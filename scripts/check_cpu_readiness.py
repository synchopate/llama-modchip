#!/usr/bin/env python3
"""
check_cpu_readiness.py — Hardware Compatibility & Performance Estimator
for Nemotron-3.5-30B-A3B (256K Context, CPU-Only + RouteCrack Speculative Engine)

Zero-dependency Python 3 script (standard library only).
Inspects CPU topology, SIMD/AMX ISA extensions, L3 cache capacity, RAM,
Transparent HugePages (THP), runs a fast multi-core DRAM bandwidth micro-benchmark,
and estimates real-world latency (TTFT) and throughput (tok/s) for interactive
autonomous coding agents (e.g. Claude Code).

Usage:
    python3 scripts/check_cpu_readiness.py [--json] [--no-bench]

Windows (PowerShell) — run from a file so the benchmark workers can be spawned:
    curl.exe -sLo check_cpu_readiness.py https://raw.githubusercontent.com/synchopate/llama-modchip/main/scripts/check_cpu_readiness.py
    python check_cpu_readiness.py
"""

import argparse
import ctypes
import ctypes.util
import glob
import json
import mmap
import multiprocessing
import os
import platform
import re
import struct
import subprocess
import sys
import time

if "__file__" in globals():
    _SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)

try:
    from _cpu_topology import (  # noqa: E402
        discover_l3_domains,
        drop_l3less_cores,
        enumerate_online_physical_cores,
        estimate_hybrid_effective_cores,
        format_cpu_list,
        parse_cpu_list,
    )
except ImportError:
    # Standalone fallback when piped via `curl -sL .../check_cpu_readiness.py | python3`
    def _read_sysfs(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def parse_cpu_list(spec):
        out = set()
        for part in (spec or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                if lo_s.isdigit() and hi_s.isdigit():
                    out.update(range(int(lo_s), int(hi_s) + 1))
            elif part.isdigit():
                out.add(int(part))
        return out

    def format_cpu_list(cpus):
        if not cpus:
            return ""
        xs = sorted(set(cpus))
        parts = []
        start = prev = xs[0]
        for c in xs[1:]:
            if c == prev + 1:
                prev = c
            else:
                parts.append(f"{start}-{prev}" if prev - start >= 2 else (f"{start},{prev}" if prev != start else f"{start}"))
                start = prev = c
        parts.append(f"{start}-{prev}" if prev - start >= 2 else (f"{start},{prev}" if prev != start else f"{start}"))
        return ",".join(parts)

    def discover_l3_domains():
        domains = {}
        for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cache/index[0-9]*")):
            lvl = _read_sysfs(os.path.join(d, "level"))
            typ = _read_sysfs(os.path.join(d, "type"))
            if lvl != "3" or typ not in ("Data", "Unified"):
                continue
            s_list = _read_sysfs(os.path.join(d, "shared_cpu_list"))
            sz_str = (_read_sysfs(os.path.join(d, "size")) or "").upper()
            if not s_list or not sz_str:
                continue
            try:
                mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}.get(sz_str[-1], 1)
                domains[s_list] = int(sz_str[:-1] if mult > 1 else sz_str) * mult
            except ValueError:
                pass
        return domains

    def enumerate_online_physical_cores(ignore_affinity=False):
        online_str = _read_sysfs("/sys/devices/system/cpu/online")
        online_set = parse_cpu_list(online_str) if online_str else set()
        aff = set()
        if not ignore_affinity:
            try:
                aff = os.sched_getaffinity(0)
            except (AttributeError, OSError):
                aff = set()
        core_map = {}
        for cpu_dir in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*")):
            m = re.search(r"cpu(\d+)$", cpu_dir)
            if not m:
                continue
            cid_num = int(m.group(1))
            if (online_set and cid_num not in online_set) or (aff and cid_num not in aff):
                continue
            pkg_s = _read_sysfs(os.path.join(cpu_dir, "topology/physical_package_id"))
            die_s = _read_sysfs(os.path.join(cpu_dir, "topology/die_id")) or "0"
            core_s = _read_sysfs(os.path.join(cpu_dir, "topology/core_id"))
            if pkg_s is not None and core_s is not None:
                try:
                    key = (int(pkg_s), int(die_s) if die_s.lstrip("-").isdigit() else 0, int(core_s))
                    if key not in core_map or cid_num < core_map[key]:
                        core_map[key] = cid_num
                except ValueError:
                    pass
        if core_map:
            return sorted(core_map.values())
        if aff:
            return sorted(aff)
        if online_set:
            return sorted(online_set)
        return list(range(os.cpu_count() or 8))

    def _cache_levels_fb(cid):
        levels = set()
        for d in glob.glob(f"/sys/devices/system/cpu/cpu{cid}/cache/index[0-9]*"):
            lvl = _read_sysfs(os.path.join(d, "level"))
            typ = _read_sysfs(os.path.join(d, "type"))
            if lvl and lvl.isdigit() and typ in ("Data", "Unified"):
                levels.add(int(lvl))
        return levels or None

    def drop_l3less_cores(phys_cpus, log=None, respect_env=True):
        cpuinfo = _read_sysfs("/proc/cpuinfo") or ""
        if (
            (respect_env and os.environ.get("MODCHIP_KEEP_NO_L3") == "1")
            or (respect_env and os.environ.get("TASKSET_MASK", "").strip() != "")
            or bool(re.search(r"\bhypervisor\b", cpuinfo))
            or len(phys_cpus) < 5
        ):
            return phys_cpus, []
        info = {c: _cache_levels_fb(c) for c in phys_cpus}
        if any(v is None for v in info.values()):
            return phys_cpus, []
        with_l3 = [c for c in phys_cpus if 3 in info[c]]
        no_l3 = [c for c in phys_cpus if 3 not in info[c] and 2 in info[c]]
        if len(with_l3) >= 4 and no_l3 and len(with_l3) + len(no_l3) == len(phys_cpus) and 2 * len(no_l3) <= len(with_l3):
            if log is not None:
                log(f"[topology] excluding L3-less cores {no_l3} (SoC-tile LP-E); set MODCHIP_KEEP_NO_L3=1 to keep them")
            return with_l3, no_l3
        return phys_cpus, []

    def estimate_hybrid_effective_cores(phys_cpus):
        n = float(len(phys_cpus) or 1)
        freqs = []
        for c in phys_cpus:
            s = _read_sysfs(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq")
            if s and s.isdigit() and int(s) > 0:
                freqs.append(int(s))
        if len(freqs) != len(phys_cpus) or not freqs:
            return n, n, n, False
        max_f, min_f = float(max(freqs)), float(min(freqs))
        if min_f >= 0.88 * max_f:
            return n, n, n, False
        is_intel = "GenuineIntel" in (_read_sysfs("/proc/cpuinfo") or "")
        weights = [(f / max_f) * (0.60 if (is_intel and (f / max_f) < 0.88) else 1.0) for f in freqs]
        dyn_c, stat_c = sum(weights), len(weights) * min(weights)
        return stat_c, dyn_c, 0.50 * stat_c + 0.50 * dyn_c, True


# ============================================================================
# 1. System & CPU Topology Detection
# ============================================================================

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=4)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def parse_size_to_mb(val_str):
    if not val_str:
        return 0.0
    m = re.search(r"([\d\.]+)\s*(KiB|MiB|GiB|KB|MB|GB|K|M|G)?", val_str, re.IGNORECASE)
    if not m:
        return 0.0
    num = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    if unit.startswith("K"):
        return num / 1024.0
    if unit.startswith("M"):
        return num
    if unit.startswith("G"):
        return num * 1024.0
    return num / (1024.0 * 1024.0)


# ----------------------------------------------------------------------------
# Native Windows detection (ctypes + winreg, standard library only).
# /proc, /sys and lscpu do not exist on Windows, so without this every field
# falls back to zero (0 GB RAM, 0 MB L3, no SIMD flags) and the verdict is wrong.
# ----------------------------------------------------------------------------

_WIN_RELATION_PROCESSOR_CORE = 0
_WIN_RELATION_NUMA_NODE = 1
_WIN_RELATION_CACHE = 2
_WIN_RELATION_PROCESSOR_PACKAGE = 3
_WIN_RELATION_ALL = 0xFFFF

# IsProcessorFeaturePresent() constants (winnt.h)
_PF_ARM_V8_INSTRUCTIONS_AVAILABLE = 29
_PF_AVX_INSTRUCTIONS_AVAILABLE = 39
_PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
_PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
_PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE = 43


def _win_cpu_registry():
    """Return (brand string, vendor id) from the CentralProcessor registry key."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as k:
            name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            try:
                vendor = str(winreg.QueryValueEx(k, "VendorIdentifier")[0]).strip()
            except OSError:
                vendor = ""
            return name, vendor
    except Exception:
        return "", ""


def _win_processor_records():
    """Raw SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records as (relationship, bytes)."""
    try:
        fn = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        fn.restype = ctypes.c_int
        size = ctypes.c_uint32(0)
        fn(_WIN_RELATION_ALL, None, ctypes.byref(size))
        if size.value == 0:
            return []
        buf = ctypes.create_string_buffer(size.value)
        if not fn(_WIN_RELATION_ALL, buf, ctypes.byref(size)):
            return []
        raw = buf.raw[: size.value]
    except Exception:
        return []
    recs = []
    off = 0
    while off + 8 <= len(raw):
        rel, rec_size = struct.unpack_from("<II", raw, off)
        if rec_size < 8:
            break
        recs.append((rel, raw[off : off + rec_size]))
        off += rec_size
    return recs


def _win_group_mask_cpus(rec, offset, count):
    """Decode a GROUP_AFFINITY[count] array into global logical CPU indices."""
    ptr = struct.calcsize("P")  # KAFFINITY is pointer-sized
    bits = ptr * 8
    cpus = []
    for i in range(max(1, count)):
        base = offset + i * (ptr + 8)  # KAFFINITY Mask; WORD Group; WORD Reserved[3]
        if base + ptr + 2 > len(rec):
            break
        mask = int.from_bytes(rec[base : base + ptr], "little")
        group = struct.unpack_from("<H", rec, base + ptr)[0]
        cpus.extend(group * bits + b for b in range(bits) if (mask >> b) & 1)
    return cpus


def _win_max_mhz():
    """Per-logical-CPU MaxMhz via CallNtPowerInformation(ProcessorInformation)."""

    class _PPI(ctypes.Structure):
        _fields_ = [
            ("Number", ctypes.c_ulong),
            ("MaxMhz", ctypes.c_ulong),
            ("CurrentMhz", ctypes.c_ulong),
            ("MhzLimit", ctypes.c_ulong),
            ("MaxIdleState", ctypes.c_ulong),
            ("CurrentIdleState", ctypes.c_ulong),
        ]

    try:
        n = os.cpu_count() or 1
        arr = (_PPI * n)()
        status = ctypes.WinDLL("powrprof").CallNtPowerInformation(11, None, 0, arr, ctypes.sizeof(arr))
        if status != 0:
            return {}
        return {int(p.Number): int(p.MaxMhz) for p in arr if p.MaxMhz > 0}
    except Exception:
        return {}


def _win_hybrid_effective_cores(cores, is_intel):
    """
    Windows analogue of estimate_hybrid_effective_cores(): cores is a list of
    (first_logical_cpu, EfficiencyClass). Windows marks P-cores with the highest
    EfficiencyClass; lower classes get the clock ratio and (on Intel) the same
    0.60 E-core vector-width factor used by the Linux path.
    """
    n = float(len(cores) or 1)
    classes = {e for _, e in cores}
    if len(classes) <= 1:
        return n, n, n, False
    top = max(classes)
    mhz = _win_max_mhz()
    freqs = [mhz.get(c) for c, _ in cores]
    max_f = float(max(freqs)) if all(freqs) else 0.0
    weights = []
    for (c, eff), f in zip(cores, freqs):
        clk = (f / max_f) if max_f > 0 else 1.0
        vec = 0.60 if (is_intel and eff < top) else 1.0
        weights.append(clk * vec)
    dyn_c, stat_c = sum(weights), len(weights) * min(weights)
    return stat_c, dyn_c, 0.50 * stat_c + 0.50 * dyn_c, True


def _win_cpu_flags(vendor, arch):
    flags = set()
    try:
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
    except Exception:
        return flags
    if arch.upper() in ("ARM64", "AARCH64"):
        if present(_PF_ARM_V8_INSTRUCTIONS_AVAILABLE):
            flags.update(["neon", "asimd"])
        if present(_PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE):
            flags.add("asimddp")
        return flags
    if present(_PF_AVX_INSTRUCTIONS_AVAILABLE):
        flags.add("avx")
    if present(_PF_AVX2_INSTRUCTIONS_AVAILABLE):
        flags.add("avx2")
    if present(_PF_AVX512F_INSTRUCTIONS_AVAILABLE):
        flags.add("avx512f")
        # Windows exposes no feature bit for VNNI/BF16/VBMI. Every AMD core with
        # AVX-512 (Zen 4, Zen 5) implements all three; on Intel we stay conservative.
        if vendor == "AuthenticAMD":
            flags.update(["avx512_vnni", "avx512_bf16", "avx512_vbmi"])
    return flags


def _detect_windows_cpu(info):
    name, vendor = _win_cpu_registry()
    if name:
        info["model_name"] = name
    info["flags"].update(_win_cpu_flags(vendor, info["arch"]))

    recs = _win_processor_records()
    cores = []  # (first logical cpu, efficiency class)
    logical = set()
    packages = numa = 0
    l3_bytes = 0
    l3_instances = 0
    for rel, rec in recs:
        if rel == _WIN_RELATION_PROCESSOR_CORE and len(rec) >= 32:
            eff = rec[9]
            group_count = struct.unpack_from("<H", rec, 30)[0]
            cpus = _win_group_mask_cpus(rec, 32, group_count)
            if cpus:
                cores.append((min(cpus), eff))
                logical.update(cpus)
        elif rel == _WIN_RELATION_PROCESSOR_PACKAGE:
            packages += 1
        elif rel == _WIN_RELATION_NUMA_NODE:
            numa += 1
        elif rel == _WIN_RELATION_CACHE and len(rec) >= 20:
            level = rec[8]
            cache_size = struct.unpack_from("<I", rec, 12)[0]
            cache_type = struct.unpack_from("<i", rec, 16)[0]  # 0=Unified, 1=Instruction, 2=Data
            if level == 3 and cache_type in (0, 2):
                l3_bytes += cache_size
                l3_instances += 1

    if cores:
        cores.sort()
        info["physical_core_ids"] = [c for c, _ in cores]
        info["physical_cores"] = len(cores)
        info["logical_threads"] = len(logical)
        stat_c, dyn_c, blend_c, is_hyb = _win_hybrid_effective_cores(cores, vendor == "GenuineIntel")
        info["is_hybrid"] = is_hyb
        info["hybrid_static_cores"] = round(stat_c, 2)
        info["hybrid_dynamic_cores"] = round(dyn_c, 2)
        info["hybrid_blended_cores"] = round(blend_c, 2)
    else:
        info["physical_core_ids"] = list(range(info["physical_cores"]))
    if packages:
        info["sockets"] = packages
    if numa:
        info["numa_nodes"] = numa
    if l3_bytes:
        info["l3_total_mb"] = l3_bytes / (1024.0 * 1024.0)
        info["l3_instances"] = l3_instances


def _detect_windows_memory(mem):
    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    try:
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            mem["total_gb"] = st.ullTotalPhys / (1024.0 ** 3)
            mem["available_gb"] = st.ullAvailPhys / (1024.0 ** 3)
    except Exception:
        pass
    mem["thp_enabled"] = "n/a"
    mem["thp_defrag"] = "n/a"


def _bw_libc():
    """C runtime exposing memset/memchr (glibc / libSystem on POSIX, msvcrt on Windows)."""
    try:
        if sys.platform == "win32":
            return ctypes.CDLL("msvcrt")
        name = ctypes.util.find_library("c")
        return ctypes.CDLL(name) if name else None
    except OSError:
        return None


def detect_cpu_and_cache():
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "model_name": platform.processor() or "Unknown CPU",
        "sockets": 1,
        "physical_cores": os.cpu_count() or 1,
        "logical_threads": os.cpu_count() or 1,
        "numa_nodes": 1,
        "l3_total_mb": 0.0,
        "l3_instances": 1,
        "flags": set(),
    }

    if info["os"] == "Linux":
        # Read /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                cpuinfo = f.read()
            for line in cpuinfo.splitlines():
                if line.startswith("model name") and info["model_name"] in ("Unknown CPU", "x86_64", ""):
                    info["model_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("flags") or line.startswith("Features"):
                    flags_str = line.split(":", 1)[1].strip().lower()
                    info["flags"].update(flags_str.split())
        except Exception:
            pass

        # Parse lscpu for accurate physical cores, sockets, L3 cache, NUMA
        lscpu_out = run_cmd("LC_ALL=C lscpu")
        cores_per_socket = None
        sockets = 1
        hypervisor = None
        for line in lscpu_out.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            kl = k.lower()
            if kl == "model name":
                info["model_name"] = v
            elif kl == "hypervisor vendor":
                hypervisor = v
            elif kl == "socket(s)":
                try:
                    sockets = int(v)
                    info["sockets"] = sockets
                except ValueError:
                    pass
            elif kl == "core(s) per socket":
                try:
                    cores_per_socket = int(v)
                except ValueError:
                    pass
            elif kl == "cpu(s)":
                try:
                    info["logical_threads"] = int(v)
                except ValueError:
                    pass
            elif kl == "numa node(s)":
                try:
                    info["numa_nodes"] = int(v)
                except ValueError:
                    pass
            elif kl == "l3 cache":
                # e.g. "64 MiB (2 instances)"
                info["l3_total_mb"] = parse_size_to_mb(v)
                inst_m = re.search(r"\((\d+)\s+instance", v, re.IGNORECASE)
                if inst_m:
                    info["l3_instances"] = int(inst_m.group(1))

        if cores_per_socket:
            info["physical_cores"] = cores_per_socket * sockets

        # KVM/QEMU often exposes fake 16MB L3 per vCPU ("16 instances = 256 MiB")
        if hypervisor and info["l3_instances"] >= 8 and info["l3_total_mb"] / info["l3_instances"] == 16.0:
            info["l3_total_mb"] = 64.0
            info["l3_instances"] = 2
            info["hypervisor"] = hypervisor

        # Fallback L3 detection via sysfs if lscpu didn't report it
        if info["l3_total_mb"] <= 0:
            l3_doms = discover_l3_domains()
            if l3_doms:
                info["l3_total_mb"] = sum(l3_doms.values()) / (1024.0 * 1024.0)
                info["l3_instances"] = len(l3_doms)

        # Enumerate exact online non-SMT physical core IDs & exclude minority L3-less LP-E cores
        env_mask = os.environ.get("TASKSET_MASK", "").strip()
        if env_mask:
            parsed_mask = sorted(parse_cpu_list(env_mask))
            phys_ids = parsed_mask if parsed_mask else enumerate_online_physical_cores()
            dropped_lpe = []
        else:
            phys_ids = enumerate_online_physical_cores()
            phys_ids, dropped_lpe = drop_l3less_cores(
                phys_ids, log=lambda msg: sys.stderr.write(msg + "\n")
            )
        info["physical_core_ids"] = phys_ids
        info["physical_cores"] = len(phys_ids)
        info["dropped_no_l3_cores"] = dropped_lpe
        stat_c, dyn_c, blend_c, is_hyb = estimate_hybrid_effective_cores(phys_ids)
        info["is_hybrid"] = is_hyb
        info["hybrid_static_cores"] = round(stat_c, 2)
        info["hybrid_dynamic_cores"] = round(dyn_c, 2)
        info["hybrid_blended_cores"] = round(blend_c, 2)

    elif info["os"] == "Darwin":
        model = run_cmd("sysctl -n machdep.cpu.brand_string")
        if model:
            info["model_name"] = model
        pcores = run_cmd("sysctl -n hw.physicalcpu")
        lcores = run_cmd("sysctl -n hw.logicalcpu")
        if pcores.isdigit():
            info["physical_cores"] = int(pcores)
        if lcores.isdigit():
            info["logical_threads"] = int(lcores)
        l3 = run_cmd("sysctl -n hw.l3cachesize")
        if l3.isdigit() and int(l3) > 0:
            info["l3_total_mb"] = int(l3) / (1024.0 * 1024.0)
        else:
            # Apple Silicon SLC / L2 estimate
            l2 = run_cmd("sysctl -n hw.l2cachesize")
            if l2.isdigit() and int(l2) > 0:
                info["l3_total_mb"] = (int(l2) * 2) / (1024.0 * 1024.0)
        if info["arch"] == "arm64":
            info["flags"].update(["neon", "asimd", "dotprod", "apple_amx"])
        info["physical_core_ids"] = list(range(info["physical_cores"]))
    elif info["os"] == "Windows":
        _detect_windows_cpu(info)
    else:
        info["physical_core_ids"] = list(range(info["physical_cores"]))

    phys_ids = info.get("physical_core_ids", list(range(info["physical_cores"])))
    env_mask_override = os.environ.get("TASKSET_MASK", "").strip() if info["os"] == "Linux" else ""
    info["taskset_mask"] = env_mask_override if env_mask_override else (format_cpu_list(phys_ids) or "0")

    return info


def detect_memory_and_thp():
    mem = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "thp_enabled": "unknown",
        "thp_defrag": "unknown",
        "shm_total_gb": 0.0,
    }
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem["total_gb"] = int(line.split()[1]) / (1024.0 * 1024.0)
                    elif line.startswith("MemAvailable:"):
                        mem["available_gb"] = int(line.split()[1]) / (1024.0 * 1024.0)
        except Exception:
            pass

        try:
            with open("/sys/kernel/mm/transparent_hugepage/enabled", "r", encoding="utf-8") as f:
                s = f.read().strip()
                m = re.search(r"\[(\w+)\]", s)
                mem["thp_enabled"] = m.group(1) if m else s
        except Exception:
            pass

        try:
            with open("/sys/kernel/mm/transparent_hugepage/defrag", "r", encoding="utf-8") as f:
                s = f.read().strip()
                m = re.search(r"\[([^\]]+)\]", s)
                mem["thp_defrag"] = m.group(1) if m else s
        except Exception:
            pass

        try:
            st = os.statvfs("/dev/shm")
            mem["shm_total_gb"] = (st.f_blocks * st.f_frsize) / (1024.0 ** 3)
        except Exception:
            pass

        # Handle un-capped LXC / container cgroups that report 1 PiB (1048576 GiB) MemTotal
        if mem["total_gb"] > 65536.0:
            numa_kb = 0
            numa_free_kb = 0
            for npath in glob.glob("/sys/devices/system/node/node*/meminfo"):
                try:
                    with open(npath, "r", encoding="utf-8") as nf:
                        for nline in nf:
                            if "MemTotal:" in nline:
                                numa_kb += int(nline.split()[-2])
                            elif "MemFree:" in nline:
                                numa_free_kb += int(nline.split()[-2])
                except Exception:
                    pass
            if 0 < numa_kb < (65536 * 1024 * 1024):
                mem["total_gb"] = numa_kb / (1024.0 * 1024.0)
                mem["available_gb"] = min(mem["available_gb"], max(numa_free_kb / (1024.0 * 1024.0), mem["total_gb"] * 0.80))
            elif mem["shm_total_gb"] > 0.5:
                mem["total_gb"] = mem["shm_total_gb"] if mem["shm_total_gb"] >= 24.0 else mem["shm_total_gb"] * 2.0
                mem["available_gb"] = min(mem["available_gb"], mem["total_gb"] * 0.85)
    elif platform.system() == "Darwin":
        tot = run_cmd("sysctl -n hw.memsize")
        if tot.isdigit():
            mem["total_gb"] = int(tot) / (1024.0 ** 3)
            mem["available_gb"] = mem["total_gb"] * 0.75
        mem["thp_enabled"] = "macos_superpages"
    elif platform.system() == "Windows":
        _detect_windows_memory(mem)

    return mem


# ============================================================================
# 2. Multi-Core Sustained DRAM Read Bandwidth Micro-Benchmark (Zero-Dependency)
# ============================================================================

_BW_READY = None
_BW_START_EVT = None


def _bw_pool_init(ready_val, start_evt):
    global _BW_READY, _BW_START_EVT
    _BW_READY = ready_val
    _BW_START_EVT = start_evt


def _bw_worker(args):
    buf_mb, iters = args
    size = buf_mb * 1024 * 1024
    libc = _bw_libc()

    # Allocate anonymous mmap buffer and fault all pages in BEFORE barrier
    src = mmap.mmap(-1, size)

    if libc:
        src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(src))
        libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        libc.memchr.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        libc.memchr.restype = ctypes.c_void_p
        libc.memset(src_ptr, 0x5A, size)
    else:
        src[:] = b"\x5a" * size

    if _BW_READY is not None:
        with _BW_READY.get_lock():
            _BW_READY.value += 1
    if _BW_START_EVT is not None:
        _BW_START_EVT.wait()

    t0 = time.perf_counter()
    if libc:
        # Pure SIMD DRAM Read scan (AVX-512 / AVX2 / NEON in glibc memchr):
        # Avoids the 2:1 Infinity Fabric GMI3 write-throttling penalty on 2-CCD Ryzen CPUs
        for _ in range(iters):
            libc.memchr(src_ptr, 0xFF, size)
    else:
        for _ in range(iters):
            src.find(b"\xff")
    elapsed = time.perf_counter() - t0
    src.close()
    bytes_read = float(size) * iters
    return bytes_read, elapsed


def measure_dram_bandwidth_gbs(physical_cores, total_ram_gb):
    # Use up to 16 workers, 128 MB per worker (pure SIMD read stream after page-fault barrier)
    n_workers = min(max(1, physical_cores), 16)
    buf_mb = 64 if total_ram_gb < 16 else 128
    iters = 6

    try:
        # Both paths use the same ready-barrier so that interpreter start-up (slow under
        # 'spawn', e.g. on Windows) and page-faulting are excluded from the timed window.
        use_fork = "fork" in multiprocessing.get_all_start_methods()
        if not use_fork:
            # 'spawn' children re-import __main__ from disk. When the script is piped via
            # stdin there is no file to import: workers crash and the Pool respawns them
            # forever. Skip the benchmark (caller falls back to a default bandwidth).
            main_file = getattr(sys.modules.get("__main__"), "__file__", None)
            if not main_file or not os.path.isfile(main_file):
                return 0.0
        ctx = multiprocessing.get_context("fork" if use_fork else "spawn")
        ready_timeout = 4.0 if use_fork else 20.0
        ready = ctx.Value("i", 0)
        evt = ctx.Event()
        pool = ctx.Pool(processes=n_workers, initializer=_bw_pool_init, initargs=(ready, evt))
        async_res = [pool.apply_async(_bw_worker, ((buf_mb, iters),)) for _ in range(n_workers)]
        t_wait0 = time.perf_counter()
        while ready.value < n_workers and (time.perf_counter() - t_wait0) < ready_timeout:
            time.sleep(0.005)
        t0 = time.perf_counter()
        evt.set()
        results = [r.get(timeout=30) for r in async_res]
        wall_time = time.perf_counter() - t0
        pool.close()
        pool.join()

        total_bytes = sum(b for b, _ in results)
        max_worker_time = max(t for _, t in results)
        eff_time = max(max_worker_time, wall_time * 0.90, 1e-4)
        return (total_bytes / 1e9) / eff_time
    except Exception:
        return 0.0


# ============================================================================
# 3. Analytical Model for Nemotron-3.5-30B-A3B (256K Context + RouteCrack)
# ============================================================================

def evaluate_readiness(cpu, mem, measured_bw_gbs):
    flags = cpu["flags"]
    has_amx = "amx_int8" in flags or "amx_tile" in flags
    has_avx512_vnni = "avx512_vnni" in flags
    has_avx512_bf16 = "avx512_bf16" in flags
    has_avx512_vbmi = "avx512_vbmi" in flags or "avx512vbmi" in flags
    has_avx512f = "avx512f" in flags
    has_avx_vnni = "avx_vnni" in flags
    has_avx2 = "avx2" in flags
    has_avx1 = "avx" in flags
    has_arm_neon = "asimd" in flags or "neon" in flags or "apple_amx" in flags

    # SIMD efficiency multiplier relative to Zen 5 AVX-512 VNNI (reference = 1.00)
    if has_amx and has_avx512_vnni:
        simd_tier = "Intel AMX + AVX-512 VNNI"
        decode_compute_mult = 1.12
        prefill_compute_mult = 1.85
    elif has_avx512_vnni and has_avx512_bf16:
        simd_tier = "AVX-512 VNNI + BF16 (Zen 4/5 or Xeon)"
        decode_compute_mult = 1.00
        prefill_compute_mult = 1.00
    elif has_avx512_vnni or has_avx512f:
        simd_tier = "AVX-512"
        decode_compute_mult = 0.92
        prefill_compute_mult = 0.85
    elif has_avx_vnni or has_avx2:
        simd_tier = "AVX2" + (" + AVX-VNNI" if has_avx_vnni else " (No AVX-512 VNNI)")
        # On memory-bound single-token decode, 6+ AVX2 cores sustain ~90% of AVX-512 throughput per GB/s;
        # on hybrid Intel P+E CPUs (AVX-VNNI), E-core synchronization adds minor barrier overhead (~0.85).
        is_hyb_cpu = bool(cpu.get("is_hybrid", False))
        decode_compute_mult = (0.85 if is_hyb_cpu else 0.92) if has_avx_vnni else 0.90
        prefill_compute_mult = 0.72 if has_avx_vnni else (0.58 if cpu["physical_cores"] < 20 else 0.68)
    elif has_avx1:
        simd_tier = "AVX-only (No AVX2)"
        decode_compute_mult = 0.45
        prefill_compute_mult = 0.35
    elif has_arm_neon:
        simd_tier = "ARM64 NEON / Apple Silicon"
        decode_compute_mult = 0.92
        prefill_compute_mult = 0.88
    else:
        simd_tier = "Scalar / Legacy SIMD (Unsupported for high speed)"
        decode_compute_mult = 0.25
        prefill_compute_mult = 0.20

    # Calibrate effective pure-read bandwidth against Zen 5 9955HX/9950X reference (~50.5 GB/s pure read)
    ref_memcpy_bw = 50.5
    eff_bw = measured_bw_gbs if measured_bw_gbs > 5.0 else 38.0
    bw_ratio = eff_bw / ref_memcpy_bw

    # L3 cache residency bonus/penalty:
    # Active MoE experts (~1.7 GB/token) stream from DRAM regardless of 8-64 MB L3, while EAKV + Mamba-2
    # keep hot state small; only 128-256+ MB LLCs cache hot shared/routed experts across tokens.
    l3_mb = cpu["l3_total_mb"]
    if cpu.get("os") == "Darwin" and cpu.get("arch") == "arm64":
        l3_status = "APPLE SILICON UNIFIED MEMORY + SLC (macOS hides SLC size)"
        l3_mult = 1.00
    elif l3_mb >= 256.0:
        l3_status = f"FULL 256K KV + HOT EXPERTS IN L3 ({l3_mb:.0f} MB LLC)"
        l3_mult = 1.12
    elif l3_mb >= 120.0:
        l3_status = f"MULTI-CCD HIGH-CAPACITY L3 RESIDENCY ({l3_mb:.0f} MB L3)"
        l3_mult = 1.03
    elif l3_mb >= 60.0:
        l3_status = f"OPTIMAL SWA + MAMBA-2 L3 RESIDENCY ({l3_mb:.0f} MB L3)"
        l3_mult = 1.00
    elif l3_mb >= 24.0:
        l3_status = f"PARTIAL L3 RESIDENCY ({l3_mb:.0f} MB L3 — recommend LLAMA_ATTN_WINDOW=4096)"
        l3_mult = 0.97
    else:
        l3_status = f"SMALL L3 ({l3_mb:.0f} MB — high L3 eviction pressure)"
        l3_mult = 0.95

    # Core scaling factor for decode & hybrid-aware static/dynamic bounds for prefill
    decode_core_ratio = min(cpu["physical_cores"] / 16.0, 4.0)
    is_hybrid = bool(cpu.get("is_hybrid", False))
    stat_cores = float(cpu.get("hybrid_static_cores", cpu["physical_cores"]))
    dyn_cores = float(cpu.get("hybrid_dynamic_cores", cpu["physical_cores"]))
    stat_core_ratio = min(stat_cores / 16.0, 4.0)
    dyn_core_ratio = min(dyn_cores / 16.0, 4.0)

    # Combined decode multiplier:
    # When measured_bw_gbs > 5.0, the benchmark already ran on cpu["physical_cores"] workers,
    # so bw_ratio already reflects core-count memory-level parallelism (only <6 cores need a small MLP factor).
    if measured_bw_gbs > 5.0:
        core_mlp = min(1.0, 0.70 + 0.05 * cpu["physical_cores"])
        bw_core_term = bw_ratio * core_mlp
    else:
        bw_core_term = 0.75 * bw_ratio + 0.25 * min(decode_core_ratio, 1.5)
    decode_scale = bw_core_term * decode_compute_mult * l3_mult
    if cpu.get("os") == "Darwin" and cpu.get("arch") == "arm64":
        # Apple CPU clusters hit a NEON dequant+dot compute ceiling well below SoC fabric bandwidth;
        # calibrated to #8 (M4 Max 16C @ 238.9 GB/s -> 49.2-57.7 tok/s prose/code, 70.9-120 agent/refactor).
        decode_scale = min(decode_scale, 2.7 * min(cpu["physical_cores"] / 16.0, 1.25))
    # Cap unrealistic extrapolations
    decode_scale = max(0.15, min(decode_scale, 5.5))

    bw_prefill_factor = min(max(bw_ratio, 0.5), 2.5)
    prefill_stat_scale = max(0.15, min((stat_core_ratio ** 0.85) * prefill_compute_mult * bw_prefill_factor, 6.0))
    prefill_dyn_scale = max(0.15, min((dyn_core_ratio ** 0.85) * prefill_compute_mult * bw_prefill_factor, 6.0))
    prefill_scale = prefill_stat_scale if is_hybrid else prefill_dyn_scale

    # THP check:
    # On Linux, 'always' backs anonymous buffers (~17-18.5 GiB) with 2MB THP automatically;
    # on bare-metal short context the gain is small (~0-3%, larger in VMs / long context).
    is_linux = cpu.get("os") == "Linux"
    if not is_linux:
        thp_optimal = True
        thp_ok = True
        thp_mult = 1.00
    else:
        thp_optimal = mem["thp_enabled"] == "always"
        thp_ok = mem["thp_enabled"] in ("always", "madvise")
        thp_mult = 1.00 if thp_optimal else 0.99

    base_floor_tps = 17.2 * decode_scale * thp_mult
    prose_code_tps = 20.2 * decode_scale * thp_mult
    tool_json_tps = 24.8 * decode_scale * thp_mult
    boilerplate_tps = 42.0 * decode_scale * thp_mult
    prefill_tps = 205.0 * prefill_scale * thp_mult
    prefill_stat_tps = 205.0 * prefill_stat_scale * thp_mult
    prefill_dyn_tps = 205.0 * prefill_dyn_scale * thp_mult
    warm_ttft_s = max(0.12, 0.46 / max(prefill_scale * thp_mult, 0.2))
    slot_restore_ms = max(6.0, 18.9 / max(bw_ratio, 0.3))

    # Memory capacity & multi-agent slots
    # Model weights Q4_0 = ~16.8 GB (mmap), OS + runtime overhead = ~2.0 GB
    # Each 256K full-attention slot = ~0.43 GB (or 0.14 GB with SWA=4096)
    ram_total = mem["total_gb"]
    ram_avail = mem["available_gb"]
    can_fit_model_in_ram = ram_total >= 22.0
    if ram_total >= 28.0:
        free_for_slots = max(1.0, ram_total - 20.0)
        max_256k_slots = max(1, min(int(free_for_slots / 1.2), cpu["physical_cores"] // 4 or 1))
    elif ram_total >= 22.0:
        max_256k_slots = 1
    else:
        max_256k_slots = 0

    # Overall verdict
    if not can_fit_model_in_ram:
        tier = "TIER D — INSUFFICIENT RAM (< 22 GB Total RAM)"
        summary = "Your system has less than 22 GB of RAM. The 17 GB Q4_0 weights + KV cache will swap to disk."
    elif tool_json_tps >= 45.0:
        if cpu.get("os") == "Darwin" and cpu.get("arch") == "arm64":
            tier = "TIER S — APPLE SILICON HIGH-BANDWIDTH UNIFIED MEMORY (METAL + NEON)"
            summary = "High-bandwidth Unified Memory detected; './modchip setup' enables zero-copy Metal GPU + NEON SDOT."
        else:
            tier = "TIER S — SERVER / WORKSTATION CLASS (ULTRA-FAST CPU INFERENCE)"
            summary = "Multi-channel DDR5 / large LLC detected. Outperforms reference Zen 5 testbed by a wide margin."
    elif prose_code_tps >= 16.0:
        tier = "TIER A — PRODUCTION-READY INTERACTIVE CODING AGENT (CLAUDE CODE)"
        summary = "Full 256K context & sub-second warm TTFT with smooth 20-25+ tok/s autonomous coding & tool calls."
    elif prose_code_tps >= 9.0:
        tier = "TIER B — USABLE LOCAL ASSISTANT (MODERATE SPEED)"
        summary = "Usable for local chat and background coding tasks; enable LLAMA_ATTN_WINDOW=4096 for best speed."
    else:
        tier = "TIER C — ENTRY-LEVEL / BANDWIDTH-BOTTLENECKED"
        summary = "Runs functionally, but dual-channel DDR4 or lack of AVX-512 VNNI limits interactive responsiveness."

    return {
        "simd_tier": simd_tier,
        "has_avx512_vnni": has_avx512_vnni,
        "has_amx": has_amx,
        "l3_status": l3_status,
        "thp_ok": thp_ok,
        "thp_optimal": thp_optimal,
        "measured_bw_gbs": round(eff_bw, 1),
        "can_fit_model_in_ram": can_fit_model_in_ram,
        "max_256k_slots": max_256k_slots,
        "estimates": {
            "warm_ttft_sec": round(warm_ttft_s, 2),
            "slot_restore_ms": round(slot_restore_ms, 1),
            "prefill_tok_s": round(prefill_tps, 1),
            "prefill_static_tok_s": round(prefill_stat_tps, 1),
            "prefill_dynamic_tok_s": round(prefill_dyn_tps, 1),
            "base_floor_tok_s": round(base_floor_tps, 1),
            "prose_and_code_tok_s": round(prose_code_tps, 1),
            "tool_call_json_tok_s": round(tool_json_tps, 1),
            "repetitive_code_tok_s": round(boilerplate_tps, 1),
        },
        "tier": tier,
        "summary": summary,
    }


# ============================================================================
# 4. CLI Report Formatter
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Check CPU readiness for Nemotron-30B-A3B 256K + RouteCrack")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--no-bench", action="store_true", help="Skip the 0.6s DRAM bandwidth micro-benchmark")
    args = parser.parse_args()

    # Legacy Windows code pages (cp1252/cp437) cannot encode the report's bullets/arrows;
    # degrade to '?' instead of crashing halfway through the report.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    cpu = detect_cpu_and_cache()
    mem = detect_memory_and_thp()

    if not args.json:
        print("=" * 78)
        print(" NEMOTRON-3.5-30B-A3B (256K CPU + ROUTECRACK) — HARDWARE READINESS AUDIT")
        print("=" * 78)
        print(" Running multi-core DRAM streaming micro-benchmark (~0.6s)..." if not args.no_bench else "")

    bw_gbs = 0.0 if args.no_bench else measure_dram_bandwidth_gbs(cpu["physical_cores"], mem["total_gb"])
    rep = evaluate_readiness(cpu, mem, bw_gbs)

    if args.json:
        out = {
            "cpu": {k: (list(v) if isinstance(v, set) else v) for k, v in cpu.items()},
            "memory": mem,
            "assessment": rep,
        }
        print(json.dumps(out, indent=2))
        return

    e = rep["estimates"]
    print(f"\n [1] HARDWARE TOPOLOGY & ISA EXTENSIONS")
    print(f"  • CPU Model          : {cpu['model_name']}")
    print(f"  • Physical Cores     : {cpu['physical_cores']} cores ({cpu['logical_threads']} threads, {cpu['sockets']} socket, {cpu['numa_nodes']} NUMA node, non-SMT mask: {cpu['taskset_mask']})")
    if cpu.get("dropped_no_l3_cores"):
        print(f"  • Topology Filter    : Excluded L3-less cores {cpu['dropped_no_l3_cores']} (SoC-tile LP-E; set MODCHIP_KEEP_NO_L3=1 to keep)")
    if cpu.get("is_hybrid"):
        print(f"  • Hybrid Compute Eq. : {cpu['hybrid_static_cores']:.1f} static .. {cpu['hybrid_dynamic_cores']:.1f} dynamic full-speed core eq.")
    if cpu.get("os") == "Darwin" and cpu.get("arch") == "arm64":
        print(f"  • Cache / Memory     : {rep['l3_status']} (cluster L2: {cpu['l3_total_mb']:.1f} MiB)")
    else:
        print(f"  • L3 Cache (LLC)     : {cpu['l3_total_mb']:.1f} MiB ({cpu['l3_instances']} instance(s)) -> {rep['l3_status']}")
    print(f"  • SIMD / Matrix ISA  : {rep['simd_tier']}")
    shm_note = f", /dev/shm: {mem['shm_total_gb']:.1f} GiB" if cpu.get("os") == "Linux" else ""
    print(f"  • System RAM         : {mem['total_gb']:.1f} GiB total ({mem['available_gb']:.1f} GiB available{shm_note})")
    if cpu.get("os") == "Darwin":
        print(f"  • 2MB HugePages (THP): macOS superpages (OK)")
    elif cpu.get("os") != "Linux":
        print(f"  • 2MB HugePages (THP): n/a ({cpu.get('os', 'non-Linux')})")
    else:
        thp_label = (
            "(OK)"
            if rep["thp_optimal"]
            else "(optional: set 'always' to back ~17-18.5 GiB with 2MB pages — mainly helps VMs / long context)"
        )
        print(f"  • 2MB HugePages (THP): enabled=[{mem['thp_enabled']}], defrag=[{mem['thp_defrag']}] {thp_label}")
    if bw_gbs > 0:
        print(f"  • Sustained DRAM BW  : {rep['measured_bw_gbs']:.1f} GB/s (multi-core user-space streaming)")

    ctx_note = "CPU-only NEON baseline; Metal GPU offload adds ~1.3-2.0x prefill" if (cpu.get("os") == "Darwin" and cpu.get("arch") == "arm64") else "Q4_0, 22K–35K Active Agent Context"
    print(f"\n [2] ESTIMATED PERFORMANCE ON THIS MACHINE ({ctx_note})")
    print(f"  • Slot 0 Disk Snapshot Restore (22.2K tok) : ~{e['slot_restore_ms']:.1f} ms")
    print(f"  • Warm Interactive TTFT (40-tool prefix)   : ~{e['warm_ttft_sec']:.2f} s")
    if cpu.get("is_hybrid"):
        print(f"  • Cold Prompt Prefill Throughput           : ~{e['prefill_static_tok_s']:.0f} tok/s (static bound) .. ~{e['prefill_dynamic_tok_s']:.0f} tok/s (dynamic chunks)")
    else:
        print(f"  • Cold Prompt Prefill Throughput           : ~{e['prefill_tok_s']:.0f} tok/s")
    print(f"  • Decode — Base Physical Floor (no spec)   : ~{e['base_floor_tok_s']:.1f} tok/s")
    print(f"  • Decode — Technical Prose + New Code      : ~{e['prose_and_code_tok_s']:.1f} tok/s  (Q4_0; Fractal-64 varies ~-7% to +30%)")
    print(f"  • Decode — Refactor / Tool-Call (~67% acc) : ~{e['tool_call_json_tok_s']:.1f} tok/s")
    print(f"  • Decode — High-Hit Boilerplate Burst      : ~{e['repetitive_code_tok_s']:.1f} tok/s  (90%+ PVS hit)")
    print(f"  • Max Concurrent 256K Agent Workers        : {rep['max_256k_slots']} instance(s)")

    print(f"\n [3] VERDICT & RECOMMENDED TUNING")
    print(f"  ► {rep['tier']}")
    print(f"    {rep['summary']}")

    if cpu.get("os") == "Darwin" and cpu.get("arch") == "arm64":
        print(f"\n [4] RECOMMENDED LAUNCH COMMANDS FOR APPLE SILICON (Metal + NEON SDOT)")
        print(f"  git clone https://github.com/synchopate/llama-modchip.git && cd llama-modchip")
        print(f"  ./modchip setup     # builds native arm64 + Metal (Q4_0 & Fractal-64 Q2_0 shaders)")
        print(f"  ./modchip start && ./modchip bench")
    elif cpu.get("os") == "Windows":
        core_mask = sum(1 << c for c in cpu.get("physical_core_ids", [])) or 1
        print(f"\n [4] RECOMMENDED LAUNCH FLAGS FOR THIS CPU (Windows / PowerShell)")
        print(f"  One-click GPU+CPU launcher: windows\\AVVIA_MODCHIP.bat  (see windows\\README.md)")
        print(f"  $env:LLAMA_MOE_TOP_K=4; $env:LLAMA_KV_L3_WINDOW=2048; $env:LLAMA_ATTN_WINDOW=4096; $env:LLAMA_ATTN_SINKS=16")
        print(f"  $env:LLAMA_SWA_STRIDE=1; $env:LLAMA_KV_DISABLE_CHUNKING=1; $env:OMP_PROC_BIND='close'; $env:OMP_PLACES='cores'")
        print(f"  llama-server.exe --model NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf `")
        print(f"    -c 262144 -np 1 -t {cpu['physical_cores']} -tb {cpu['physical_cores']} -b 2048 -ub 2048 --cpu-mask 0x{core_mask:X} `")
        print(f"    --cache-type-k q8_0 --cache-type-v q4_0 --repack --flash-attn auto --cache-reuse 256 --slot-save-path $env:TEMP `")
        print(f"    --ctx-checkpoints 64 --checkpoint-min-step 128 `")
        print(f"    --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 7 --spec-ngram-simple-min-hits 1")
    else:
        print(f"\n [4] RECOMMENDED LAUNCH FLAGS FOR THIS CPU (via llama_modchip.py)")
        print(f"  python3 scripts/llama_modchip.py --model /path/to/model.gguf --apply-shm --print-cmd")
        print(f"  export LLAMA_MOE_TOP_K=4 LLAMA_KV_L3_WINDOW=2048 LLAMA_ATTN_WINDOW=4096 LLAMA_ATTN_SINKS=16 LLAMA_SWA_STRIDE=1 LLAMA_KV_DISABLE_CHUNKING=1 OMP_PROC_BIND=close OMP_PLACES=cores")
        print(f"  taskset -c {cpu['taskset_mask']} llama-server --model NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf \\")
        print(f"    -c 262144 -np 1 -t {cpu['physical_cores']} -tb {cpu['physical_cores']} -b 2048 -ub 2048 \\")
        print(f"    --cache-type-k q8_0 --cache-type-v q4_0 --repack --flash-attn auto --cache-reuse 256 --slot-save-path /dev/shm \\")
        print(f"    --ctx-checkpoints 64 --checkpoint-min-step 128 \\")
        print(f"    --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 7 --spec-ngram-simple-min-hits 1")
    if cpu["numa_nodes"] > 1 or cpu["sockets"] > 1:
        cores_per_node = max(1, cpu["physical_cores"] // max(1, cpu["numa_nodes"]))
        print(f"\n  [!] Multi-NUMA / Multi-Socket Server Detected ({cpu['numa_nodes']} NUMA nodes):")
        print(f"      Avoid spanning a single 3B-active MoE across NUMA interconnects (UPI/Infinity Fabric).")
        print(f"      Instead, pin 1 independent agent instance per NUMA node:")
        print(f"      numactl --cpunodebind=0 --membind=0 llama-server ... -t {cores_per_node} --port 8789")
    if not rep["thp_optimal"] and cpu.get("os") == "Linux":
        print(f"\n  [i] Optional: Enable 2MB Transparent HugePages before launching (backs ~17-18.5 GiB with 2MB pages):")
        print(f"      echo always        | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null")
        print(f"      echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag  >/dev/null")
        print(f"      (System-wide; resets on reboot — persist via /etc/tmpfiles.d/thp.conf if desired.)")
    if (
        cpu.get("os") == "Linux"
        and cpu.get("arch") == "x86_64"
        and not cpu.get("hypervisor")
        and cpu.get("sockets", 1) == 1
        and bw_gbs > 0.0
    ):
        chassis = ""
        try:
            with open("/sys/class/dmi/id/chassis_type", "r", encoding="utf-8") as cf:
                chassis = cf.read().strip()
        except OSError:
            pass
        model_str = cpu.get("model_name", "")
        is_mobile_sku = (not re.search(r"\b(EPYC|Xeon|Threadripper)\b", model_str, re.I)) and bool(
            re.search(r"\b\d{3,5}(?:U|H|HS|HX|HX3D|P|G\d|V)\b", model_str) or re.search(r"\bRyzen AI\b", model_str, re.I)
        )
        is_portable = chassis in ("8", "9", "10", "14", "30", "31", "32", "35", "36") or is_mobile_sku
        bw_warn_limit = 25.0 if is_portable else 31.5
        if rep["measured_bw_gbs"] <= bw_warn_limit:
            print(f"\n  [i] Measured DRAM bandwidth is low ({rep['measured_bw_gbs']:.1f} GB/s).")
            if is_portable:
                print(f"      On laptops / mini-PCs, check whether only one memory channel is populated or CPU power is capped:")
                print(f"        sudo dmidecode -t memory | grep -iE \"locator|size|speed\"")
                print(f"      Ignore this if dual-channel RAM is already running at its rated speed on AC power.")
            else:
                print(f"      On desktops this often means RAM at JEDEC default (DDR4-2133/2400) instead of its")
                print(f"      XMP/DOCP/EXPO rating, or only one channel populated. Check:")
                print(f"        sudo dmidecode -t memory | grep -iE \"locator|size|speed\"")
                print(f"      Ignore this if your RAM already runs at its rated speed.")
                print(f"      (In Issue #17, decode scaled ~+45-50% from 2133 -> 3200 MT/s.)")
    if (
        cpu.get("os") == "Windows"
        and cpu.get("arch", "").upper() in ("AMD64", "X86_64")
        and cpu.get("sockets", 1) == 1
        and bw_gbs > 0.0
    ):
        model_str = cpu.get("model_name", "")
        is_mobile_sku = (not re.search(r"\b(EPYC|Xeon|Threadripper)\b", model_str, re.I)) and bool(
            re.search(r"\b\d{3,5}(?:U|H|HS|HX|HX3D|P|G\d|V)\b", model_str) or re.search(r"\bRyzen AI\b", model_str, re.I)
        )
        if rep["measured_bw_gbs"] <= (25.0 if is_mobile_sku else 31.5):
            print(f"\n  [i] Measured DRAM bandwidth is low ({rep['measured_bw_gbs']:.1f} GB/s).")
            print(f"      Check that both memory channels are populated and RAM runs at its rated speed")
            print(f"      (XMP/EXPO on desktops, AC power on laptops). In PowerShell:")
            print(f"        Get-CimInstance Win32_PhysicalMemory | Format-Table BankLabel,Capacity,ConfiguredClockSpeed")
            print(f"      A single listed module usually means single-channel operation (~half the bandwidth).")
    print("=" * 78)


if __name__ == "__main__":
    main()
