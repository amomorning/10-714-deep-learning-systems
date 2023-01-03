import io, os, json

total_markdown = 0
total_code = 0

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb") and not file.endswith("checkpoint.ipynb") :
            #print(os.path.join(root, file))
            with io.open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                nb = json.load(f)

            word_count_markdown = 0
            word_count_code = 0
            for cell in nb['cells']:
                if cell['cell_type'] == "markdown":
                    for line in cell['source']:
                        word_count_markdown += len(line.replace('#', '').lstrip().split(' '))
                elif cell['cell_type'] == "code":
                    for line in cell['source']:
                        word_count_code += len(line.replace('#', '').lstrip().split(' '))
            total_markdown += word_count_markdown
            total_code += word_count_code

print("{} Words in notebooks' markdown" .format(total_markdown))
print("{} Words in notebooks' code" .format(total_code))
