import os, re

#tag_map = {
#	"fila": "rows"
#}

#tag_map = {
#	"col": "fila"
#}

#tag_map = {
#	"rows": "col"
#}

for root, dirs, files in os.walk(r"/home/luiireye/Documents/datasets/vision-02R/"):
    for file in files:
        for tag in tag_map.keys():
            if re.search(tag, file) !=None:
                new_tag = re.sub(tag, tag_map[tag],file)
                try:
                    os.rename(os.path.join(root,file),os.path.join(root, new_tag))
                except OSError:
                    print("Desconocido")