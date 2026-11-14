"""Build a small synthetic pcap (no network needed) to test flow_tracker.py."""
from scapy.all import DNS, DNSQR, DNSRR, IP, TCP, UDP, Ether, Raw, wrpcap

pk, t = [], 1000.0
def add(p, dt):
    global t
    t += dt; p.time = t; pk.append(p)

c, s = "10.0.0.5", "10.0.0.9"
# 1) TCP flow: handshake, 2 data packets each way, FIN/FIN
add(Ether()/IP(src=c, dst=s)/TCP(sport=40000, dport=80, flags="S", window=8192), 0.0)
add(Ether()/IP(src=s, dst=c)/TCP(sport=80, dport=40000, flags="SA", window=65535), 0.010)
add(Ether()/IP(src=c, dst=s)/TCP(sport=40000, dport=80, flags="A"), 0.010)
add(Ether()/IP(src=c, dst=s)/TCP(sport=40000, dport=80, flags="PA")/Raw(b"A"*100), 0.010)
add(Ether()/IP(src=s, dst=c)/TCP(sport=80, dport=40000, flags="PA")/Raw(b"B"*500), 0.020)
add(Ether()/IP(src=c, dst=s)/TCP(sport=40000, dport=80, flags="PA")/Raw(b"A"*200), 0.010)
add(Ether()/IP(src=s, dst=c)/TCP(sport=80, dport=40000, flags="A")/Raw(b"B"*300), 0.010)
add(Ether()/IP(src=c, dst=s)/TCP(sport=40000, dport=80, flags="FA"), 0.010)
add(Ether()/IP(src=s, dst=c)/TCP(sport=80, dport=40000, flags="FA"), 0.010)
# 2) UDP DNS query + response
add(Ether()/IP(src=c, dst="8.8.8.8")/UDP(sport=5353, dport=53)/DNS(rd=1, qd=DNSQR(qname="example.com")), 0.5)
add(Ether()/IP(src="8.8.8.8", dst=c)/UDP(sport=53, dport=5353)/DNS(qr=1, qd=DNSQR(qname="example.com"),
    an=DNSRR(rrname="example.com", rdata="93.184.216.34")), 0.030)
# 3) SYN scan: 5 ports on one host, each probe answered with RST
for i, port in enumerate([21, 22, 23, 25, 8080]):
    add(Ether()/IP(src="10.0.0.66", dst=s)/TCP(sport=50000 + i, dport=port, flags="S"), 0.001)
    add(Ether()/IP(src=s, dst="10.0.0.66")/TCP(sport=port, dport=50000 + i, flags="RA"), 0.001)
wrpcap("test.pcap", pk)
print(len(pk), "packets written to test.pcap")