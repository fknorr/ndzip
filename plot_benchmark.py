from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import csv as csv_pkg


ALGOS = [
    ('lz4', '#7BCA74'),
    ('lzma', '#4DB344'),
    ('fpzip', '#5193CF'),
    ('fpc 10', '#3379BA'),
    ('fpc 30', '#2C6AA2'),
    ('hcde', '#D15757'),
    ('ndzip', '#CA3F3F'),
]

with open('/data/scidata/scidata.csv') as f:
    file_dims = dict((f, len(d.split())) for f, _, d in csv_pkg.reader(f, delimiter=';'))

data = defaultdict(lambda: defaultdict(dict))

with open(sys.argv[1]) as csv:
    try:
        while True:
            title = next(csv).rstrip()
            file_names = next(csv).rstrip().split(';')[1:]
            while True:
                l = next(csv).rstrip()
                if not l:
                    break
                row = l.split(';')
                algo = row[0]
                for i, value in enumerate(row[1:]):
                    data[title][algo][file_names[i]] = float(value)
    except StopIteration:
        pass

fig, ax = plt.subplots(nrows=len(data))

for i, (title, algo_data) in enumerate(data.items()):
    bar_width = 1/(len(algo_data)+1)
    for j, (algo, color) in enumerate(ALGOS):
        file_data = algo_data[algo]
        ax[i].bar([i+bar_width*(j-(len(algo_data)-1)/2) for i in range(len(file_data))],
                [file_data[f] for f in file_names], width=bar_width, color=color,
                label=algo)
    ax[i].set_xticks(list(range(len(file_names))))
    ax[i].set_xticklabels(['{} ({}d)'.format(f.split('.')[0], file_dims[f]) for f in file_names],
            rotation=15, ha='center')
    ax[i].legend()

    if title == 'times':
        ax[i].set_yscale('log')
        ax[i].set_ylabel('time (s)')
    else:
        ax[i].set_ylabel('compression ratio')

plt.show()

