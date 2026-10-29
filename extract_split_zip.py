"""Gabungkan dan ekstrak arsip ZIP multipart tanpa paket tambahan."""
import argparse
import mmap
import shutil
import struct
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Bagian terakhir: dataset1.zip")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    final = args.archive.resolve()
    parts = sorted(final.parent.glob(final.stem + ".z[0-9][0-9]")) + [final]
    if not final.is_file() or len(parts) < 2:
        raise SystemExit("Bagian arsip multipart tidak ditemukan.")
    for index, part in enumerate(parts[:-1], 1):
        if part.suffix != f".z{index:02d}":
            raise SystemExit(f"Bagian hilang sebelum {part.name}.")
    combined = final.with_name(final.stem + "_combined.zip")
    if combined.exists():
        raise SystemExit(f"File {combined} sudah ada; pindahkan atau hapus setelah diperiksa.")
    bases, size = [], 0
    with combined.open("xb") as out:
        for part in parts:
            bases.append(size)
            size += part.stat().st_size
            print(f"Menggabungkan {part.name}...", flush=True)
            with part.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    with combined.open("r+b") as f, mmap.mmap(f.fileno(), 0) as data:
        tail = data[max(0, len(data)-65557):]
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise ValueError("Akhir ZIP tidak ditemukan.")
        eocd += max(0, len(data)-65557)
        disk, cd_disk, _, entries, cd_size, cd_offset, comment = struct.unpack_from("<HHHHIIH", data, eocd + 4)
        if eocd + 22 + comment != len(data) or disk != len(parts)-1 or cd_disk >= len(parts):
            raise ValueError("Bagian ZIP tidak lengkap atau struktur tidak dikenali.")
        if entries == 0xffff or cd_offset == 0xffffffff or cd_size == 0xffffffff:
            raise ValueError("ZIP64 tidak didukung oleh alat ini.")
        start = bases[cd_disk] + cd_offset
        pos = start
        for _ in range(entries):
            if data[pos:pos+4] != b"PK\x01\x02":
                raise ValueError("Struktur central directory ZIP tidak dikenali.")
            name_len, extra_len, comment_len, disk_start = struct.unpack_from("<HHHH", data, pos+28)
            local_offset = struct.unpack_from("<I", data, pos+42)[0]
            if disk_start >= len(parts) or local_offset == 0xffffffff:
                raise ValueError("Disk tidak ditemukan atau ZIP64 tidak didukung.")
            adjusted = bases[disk_start] + local_offset
            if adjusted > 0xffffffff:
                raise ValueError("Ukuran memerlukan ZIP64 yang tidak didukung.")
            struct.pack_into("<H", data, pos+34, 0)
            struct.pack_into("<I", data, pos+42, adjusted)
            pos += 46 + name_len + extra_len + comment_len
        if pos != start + cd_size:
            raise ValueError("Ukuran central directory tidak cocok.")
        struct.pack_into("<HHHHII", data, eocd+4, 0, 0, entries, entries, cd_size, start)
    print("Memeriksa integritas arsip; ini dapat memakan waktu...", flush=True)
    with zipfile.ZipFile(combined) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Pemeriksaan CRC gagal: {bad}")
        args.destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(args.destination)
        print(f"Selesai: {len(archive.infolist())} entri diekstrak ke {args.destination.resolve()}")


if __name__ == "__main__":
    main()
