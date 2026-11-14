"""Scapy-based flow tracker for NetGuard.

Groups packets into bidirectional flows and computes CICFlowMeter-style features
(the same names as the CICIDS2017 CSV columns, with whitespace stripped).

Usage (from the src folder, venv active, Administrator prompt for live capture):
    python flow_tracker.py --pcap test.pcap --out flows.csv
    python flow_tracker.py --live --seconds 30 --out live_flows.csv
    python flow_tracker.py --list-ifaces
"""
import argparse
import math
from dataclasses import dataclass, field

import pandas as pd
from scapy.all import ICMP, IP, TCP, UDP, IPv6, PcapReader, conf, sniff

IDLE_TIMEOUT = 15.0     # seconds without packets -> flow is finished
MAX_DURATION = 120.0    # seconds -> long flows are cut (same idea as CICFlowMeter)

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]
META = ["src_ip", "dst_ip", "src_port", "protocol", "start_time"]


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))   # sample std


def _iats_us(times):
    return [(b - a) * 1e6 for a, b in zip(times, times[1:])]


@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    start: float
    last: float = 0.0
    fwd_t: list = field(default_factory=list)
    bwd_t: list = field(default_factory=list)
    fwd_len: list = field(default_factory=list)
    bwd_len: list = field(default_factory=list)
    fwd_hdr: int = 0
    bwd_hdr: int = 0
    fin: int = 0
    syn: int = 0
    rst: int = 0
    psh: int = 0
    ack: int = 0
    urg: int = 0
    fwd_psh: int = 0
    init_win_fwd: int = -1
    init_win_bwd: int = -1

    def add(self, ts, forward, payload, hdr, tcp):
        self.last = ts
        if forward:
            self.fwd_t.append(ts); self.fwd_len.append(payload); self.fwd_hdr += hdr
        else:
            self.bwd_t.append(ts); self.bwd_len.append(payload); self.bwd_hdr += hdr
        if tcp is not None:
            f = int(tcp.flags)
            self.fin += bool(f & 0x01); self.syn += bool(f & 0x02)
            self.rst += bool(f & 0x04); self.psh += bool(f & 0x08)
            self.ack += bool(f & 0x10); self.urg += bool(f & 0x20)
            if forward and (f & 0x08):
                self.fwd_psh += 1
            if forward and self.init_win_fwd < 0:
                self.init_win_fwd = int(tcp.window)
            if not forward and self.init_win_bwd < 0:
                self.init_win_bwd = int(tcp.window)

    def finished_by_flags(self):
        return self.rst > 0 or self.fin >= 2

    def features(self):
        all_t = sorted(self.fwd_t + self.bwd_t)
        dur_s = max(self.last - self.start, 0.0)
        dur_us = dur_s * 1e6
        n_f, n_b = len(self.fwd_len), len(self.bwd_len)
        lens = self.fwd_len + self.bwd_len
        fl, bl = self.fwd_len, self.bwd_len
        flow_iat, fwd_iat, bwd_iat = _iats_us(all_t), _iats_us(self.fwd_t), _iats_us(self.bwd_t)

        def rate(x):                       # rates are 0 for zero-length flows (no inf)
            return x / dur_s if dur_s > 0 else 0.0

        var = _std(lens) ** 2
        return {
            "Destination Port": self.dst_port,
            "Flow Duration": dur_us,
            "Total Fwd Packets": n_f, "Total Backward Packets": n_b,
            "Total Length of Fwd Packets": sum(fl), "Total Length of Bwd Packets": sum(bl),
            "Fwd Packet Length Max": max(fl, default=0), "Fwd Packet Length Min": min(fl, default=0),
            "Fwd Packet Length Mean": _mean(fl), "Fwd Packet Length Std": _std(fl),
            "Bwd Packet Length Max": max(bl, default=0), "Bwd Packet Length Min": min(bl, default=0),
            "Bwd Packet Length Mean": _mean(bl), "Bwd Packet Length Std": _std(bl),
            "Flow Bytes/s": rate(sum(lens)), "Flow Packets/s": rate(n_f + n_b),
            "Flow IAT Mean": _mean(flow_iat), "Flow IAT Std": _std(flow_iat),
            "Flow IAT Max": max(flow_iat, default=0), "Flow IAT Min": min(flow_iat, default=0),
            "Fwd IAT Total": sum(fwd_iat), "Fwd IAT Mean": _mean(fwd_iat),
            "Fwd IAT Std": _std(fwd_iat), "Fwd IAT Max": max(fwd_iat, default=0),
            "Fwd IAT Min": min(fwd_iat, default=0),
            "Bwd IAT Total": sum(bwd_iat), "Bwd IAT Mean": _mean(bwd_iat),
            "Bwd IAT Std": _std(bwd_iat), "Bwd IAT Max": max(bwd_iat, default=0),
            "Bwd IAT Min": min(bwd_iat, default=0),
            "Fwd PSH Flags": self.fwd_psh,
            "Fwd Header Length": self.fwd_hdr, "Bwd Header Length": self.bwd_hdr,
            "Fwd Packets/s": rate(n_f), "Bwd Packets/s": rate(n_b),
            "Min Packet Length": min(lens, default=0), "Max Packet Length": max(lens, default=0),
            "Packet Length Mean": _mean(lens), "Packet Length Std": _std(lens),
            "Packet Length Variance": var,
            "FIN Flag Count": self.fin, "SYN Flag Count": self.syn, "RST Flag Count": self.rst,
            "PSH Flag Count": self.psh, "ACK Flag Count": self.ack, "URG Flag Count": self.urg,
            "Down/Up Ratio": (n_b / n_f) if n_f else 0.0,
            "Average Packet Size": _mean(lens),
            "Init_Win_bytes_forward": self.init_win_fwd, "Init_Win_bytes_backward": self.init_win_bwd,
        }

    def record(self):
        row = {"src_ip": self.src_ip, "dst_ip": self.dst_ip, "src_port": self.src_port,
               "protocol": self.proto, "start_time": self.start}
        row.update(self.features())
        return row


class FlowTracker:
    def __init__(self, idle_timeout=IDLE_TIMEOUT, max_duration=MAX_DURATION):
        self.flows = {}
        self.idle_timeout, self.max_duration = idle_timeout, max_duration

    @staticmethod
    def _parse(pkt):
        if IP in pkt:
            ip = pkt[IP]; src, dst, proto = ip.src, ip.dst, ip.proto
        elif IPv6 in pkt:
            ip = pkt[IPv6]; src, dst, proto = ip.src, ip.dst, ip.nh
        else:
            return None
        tcp = pkt[TCP] if TCP in pkt else None
        if tcp is not None:
            sport, dport, payload, hdr = tcp.sport, tcp.dport, len(tcp.payload), tcp.dataofs * 4
        elif UDP in pkt:
            u = pkt[UDP]; sport, dport, payload, hdr = u.sport, u.dport, len(u.payload), 8
        elif ICMP in pkt:
            sport = dport = 0; payload, hdr = len(pkt[ICMP].payload), 8
        else:
            return None
        return src, dst, sport, dport, int(proto), payload, hdr, tcp

    def process_packet(self, pkt):
        """Add one packet. Returns a list of finished flow records (may be empty)."""
        parsed = self._parse(pkt)
        if parsed is None:
            return []
        src, dst, sport, dport, proto, payload, hdr, tcp = parsed
        ts = float(pkt.time)
        done = self.expire(ts)

        key_fwd, key_bwd = (src, sport, dst, dport, proto), (dst, dport, src, sport, proto)
        if key_fwd in self.flows:
            flow, forward = self.flows[key_fwd], True
        elif key_bwd in self.flows:
            flow, forward = self.flows[key_bwd], False
        else:
            flow = Flow(src, dst, sport, dport, proto, ts)
            self.flows[key_fwd], forward = flow, True
        flow.add(ts, forward, payload, hdr, tcp)

        if flow.finished_by_flags():
            done.append(flow.record())
            self.flows = {k: v for k, v in self.flows.items() if v is not flow}
        return done

    def expire(self, now):
        done, keep = [], {}
        for key, fl in self.flows.items():
            if now - fl.last > self.idle_timeout or now - fl.start > self.max_duration:
                done.append(fl.record())
            else:
                keep[key] = fl
        self.flows = keep
        return done

    def flush(self):
        done = [fl.record() for fl in self.flows.values()]
        self.flows = {}
        return done


def flows_from_pcap(path, **kw):
    tracker, rows = FlowTracker(**kw), []
    with PcapReader(path) as reader:
        for pkt in reader:
            rows += tracker.process_packet(pkt)
    rows += tracker.flush()
    return pd.DataFrame(rows, columns=META + FEATURES)


def flows_from_live(seconds, iface=None, **kw):
    tracker, rows = FlowTracker(**kw), []
    sniff(iface=iface, store=False, timeout=seconds,
          prn=lambda p: rows.extend(tracker.process_packet(p)))
    rows += tracker.flush()
    return pd.DataFrame(rows, columns=META + FEATURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--iface")
    ap.add_argument("--out", default="flows.csv")
    ap.add_argument("--list-ifaces", action="store_true")
    a = ap.parse_args()
    if a.list_ifaces:
        print(conf.ifaces); return
    if a.pcap:
        df = flows_from_pcap(a.pcap)
    elif a.live:
        print(f"Capturing for {a.seconds}s on {a.iface or conf.iface} ...")
        df = flows_from_live(a.seconds, a.iface)
    else:
        ap.error("give --pcap FILE or --live")
    df.to_csv(a.out, index=False)
    print(f"{len(df)} flows -> {a.out}")
    print(df[["src_ip", "dst_ip", "Destination Port", "Total Fwd Packets",
              "Total Backward Packets", "Flow Duration"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()