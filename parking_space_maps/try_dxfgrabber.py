import dxfgrabber

dxf = dxfgrabber.readfile("Ruhender_Verkehr_Urfahr.dxf")
print("DXF version: {}".format(dxf.dxfversion))
header_var_count = len(dxf.header) # dict of dxf header vars
layer_count = len(dxf.layers) # collection of layer definitions
block_definition_count = len(dxf.blocks) #  dict like collection of block definitions
entity_count = len(dxf.entities) # list like collection of entities

print(header_var_count)
print(layer_count)
print(block_definition_count)
print(entity_count)
print(dxf.header)
print(dxf.layers)
print(dxf.blocks)
print(dxf.entities)

for entity in dxf.entities:
    print(entity)
    print(entity.__dict__)
    attribs = entity.__dict__.get('attribs', [])
    for attrib in attribs:
        print(attrib.__dict__)
    print('')

