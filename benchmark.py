import csv
import os
import sys
import subprocess
import time
import re
from collections import defaultdict
from operator import itemgetter


def time_ndzip(file_name, dims):
    duration = 1e9
    for _ in range(10):
        start = time.time()
        output = subprocess.run(['./compress_cpu', os.path.join('/data/scidata', file_name),
            os.path.join('/tmp', file_name), *dims.split()], capture_output=True)
        end = time.time()
        duration = min(duration, end-start)
    try:
        ratio = float(re.search('ratio ([0-9.]+)', output.stdout.decode()).group(1))
    except:
        print(ratio)
        raise
    return duration, ratio

def time_hcde(file_name, dims):
    duration = 1e9
    for _ in range(10):
        start = time.time()
        output = subprocess.run(['/home/fabian/Code/hcde/build/release/compress', '-i', os.path.join('/data/scidata', file_name),
            '-o', os.path.join('/tmp', file_name), '-n', *dims.split(), '-e', 'cpu'], capture_output=True)
        end = time.time()
        duration = min(duration, end-start)
    ratio = float(re.search('ratio = ([0-9.]+)', output.stderr.decode()).group(1))
    return duration, ratio

def time_lz4(file_name, dims):
    duration = 1e9
    for _ in range(10):
        start = time.time()
        output = subprocess.run(['lz4', '-f', '-1', os.path.join('/data/scidata', file_name),
            os.path.join('/tmp', file_name)], capture_output=True)
        end = time.time()
        duration = min(duration, end-start)

    in_info = os.stat(os.path.join('/data/scidata', file_name))
    out_info = os.stat(os.path.join('/tmp', file_name))
    ratio = in_info.st_size / out_info.st_size
    return duration, ratio

def time_lzma(file_name, dims):
    duration = 1e9
    for _ in range(1):
        start = time.time()
        with open(os.path.join('/tmp', file_name), 'w') as f_out:
            output = subprocess.run(['lzma', '-9', '-c', os.path.join('/data/scidata', file_name)],
                    stdout=f_out, stderr=subprocess.DEVNULL)
        end = time.time()
        duration = min(duration, end-start)

    in_info = os.stat(os.path.join('/data/scidata', file_name))
    out_info = os.stat(os.path.join('/tmp', file_name))
    ratio = in_info.st_size / out_info.st_size
    return duration, ratio

def time_zstd(file_name, dims):
    duration = 1e9
    for _ in range(1):
        start = time.time()
        output = subprocess.run(['zstd', '-f', '-19', os.path.join('/data/scidata', file_name),
            '-o', os.path.join('/tmp', file_name)], capture_output=True)
        end = time.time()
        duration = min(duration, end-start)

    in_info = os.stat(os.path.join('/data/scidata', file_name))
    out_info = os.stat(os.path.join('/tmp', file_name))
    ratio = in_info.st_size / out_info.st_size
    return duration, ratio

def time_fpzip(file_name, dims):
    duration = 1e9
    dims = dims.split()
    for _ in range(2):
        start = time.time()
        output = subprocess.run(['/home/fabian/Code/fpzip-1.3.0/build/bin/fpzip', '-' + str(len(dims)), *dims,
                '-i', os.path.join('/data/scidata', file_name), '-o', os.path.join('/tmp', file_name)],
                capture_output=True)
        end = time.time()
        duration = min(duration, end-start)

    ratio = float(re.search('ratio=([0-9.]+)', output.stderr.decode()).group(1))
    return duration, ratio

def time_fpc(file_name, dims, level):
    duration = 1e9
    dims = dims.split()
    for _ in range(1 if level > 10 else 5):
        start = time.time()
        with open(os.path.join('/data/scidata', file_name)) as f_in:
            with open(os.path.join('/tmp', file_name), 'w') as f_out:
                output = subprocess.run(['/home/fabian/Code/fpc/fpc', str(level)],
                        stdin=f_in, stdout=f_out, stderr=subprocess.DEVNULL)
        end = time.time()
        duration = min(duration, end-start)

    in_info = os.stat(os.path.join('/data/scidata', file_name))
    out_info = os.stat(os.path.join('/tmp', file_name))
    ratio = in_info.st_size / out_info.st_size
    return duration, ratio

with open('/data/scidata/scidata.csv') as f:
    rows = list(csv.reader(f, delimiter=';'))

files = []
times = defaultdict(dict)
ratios = defaultdict(dict)
for file_name, _, dims in rows:
    files.append(file_name)
    times['ndzip'][file_name], ratios['ndzip'][file_name] = time_ndzip(file_name, dims)
    times['hcde'][file_name], ratios['hcde'][file_name] = time_hcde(file_name, dims)
    times['lz4'][file_name], ratios['lz4'][file_name] = time_lz4(file_name, dims)
    times['lzma'][file_name], ratios['lzma'][file_name] = time_lzma(file_name, dims)
    times['zstd'][file_name], ratios['ndzip'][file_name] = time_ndzip(file_name, dims)
    times['fpzip'][file_name], ratios['fpzip'][file_name] = time_fpzip(file_name, dims)
    times['fpc 10'][file_name], ratios['fpc 10'][file_name] = time_fpc(file_name, dims, 10)
    times['fpc 30'][file_name], ratios['fpc 30'][file_name] = time_fpc(file_name, dims, 30)

files.sort()
print('times')
print('algorithm', *files, sep=';')
for algo, m in sorted(times.items(), key=itemgetter(0)):
    print(algo, *(m[f] for f in files), sep=';')
print()
print('ratios')
print('algorithm', *files, sep=';')
for algo, m in sorted(ratios.items(), key=itemgetter(0)):
    print(algo, *(m[f] for f in files), sep=';')

