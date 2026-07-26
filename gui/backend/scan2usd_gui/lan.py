"""Detect LAN IPv4 addresses for mobile QR upload URLs."""

from __future__ import annotations

import socket
from ipaddress import IPv4Address, ip_address


def _is_private_lan(ip: str) -> bool:
    try:
        addr = ip_address(ip)
    except ValueError:
        return False
    if not isinstance(addr, IPv4Address):
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return False
    return bool(addr.is_private)


def _udp_primary_ipv4() -> str | None:
    """Best-effort primary IPv4 via UDP connect trick (no packets sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return None


def primary_lan_ipv4() -> str | None:
    """Prefer a private LAN address; fall back to any non-loopback IPv4."""
    udp = _udp_primary_ipv4()
    if udp and _is_private_lan(udp):
        return udp
    for ip in list_lan_ipv4():
        if _is_private_lan(ip):
            return ip
    if udp:
        return udp
    ips = list_lan_ipv4()
    return ips[0] if ips else None


def list_lan_ipv4() -> list[str]:
    """List non-loopback IPv4 addresses on this host (private preferred)."""
    found: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                found.add(ip)
    except OSError:
        pass

    udp = _udp_primary_ipv4()
    if udp:
        found.add(udp)

    private = sorted(ip for ip in found if _is_private_lan(ip))
    other = sorted(ip for ip in found if ip not in private and not ip.startswith("127."))
    ordered: list[str] = []
    for ip in private + other:
        if ip not in ordered:
            ordered.append(ip)
    return ordered


def build_mobile_urls(*, port: int, token: str, path: str = "/m") -> tuple[str | None, list[str]]:
    """
    Return (preferred_url, all_urls) for the phone upload page.

    ``path`` should start with ``/`` (e.g. ``/m``).
    """
    ips = list_lan_ipv4()
    primary = primary_lan_ipv4()
    if primary and primary in ips:
        ips = [primary] + [ip for ip in ips if ip != primary]
    elif primary:
        ips = [primary] + ips

    urls = [f"http://{ip}:{port}{path}?t={token}" for ip in ips]
    preferred = urls[0] if urls else None
    local = f"http://127.0.0.1:{port}{path}?t={token}"
    if local not in urls:
        urls.append(local)
    if preferred is None:
        preferred = local
    return preferred, urls
