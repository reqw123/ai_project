"""
工具函數
"""
import socket

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def compute_body_scale(kpts):
    import numpy as np
    return float(np.linalg.norm(kpts[3] - kpts[5]))
