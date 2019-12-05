import json

f_names = ['1Q.json', '2Q.json', '3Q.json', '4Q.json', '5Q.json']

for f_name in f_names:
    with open(f_name) as fp:
        js = json.load(fp)
    with open(f_name, 'w') as n_fp:
        json.dump(js, n_fp, indent=True)