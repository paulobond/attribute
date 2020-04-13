attribute2attribute_index = {'color': {'beige':0,
                  'black':1,
                  'blue':2,
                  'brown':3,
                  'cyan':4,
                  'green':5,
                  'grey':6,
                  'orange':7,
                  'pink':8,
                  'purple':9,
                  'red':10,
                  'transparent':11,
                  'white':12,
                  'yellow':13},
        'container': {'bag':14,
                  'bidon':15,
                  'bottle':16,
                  'box':17,
                  'can':18,
                  'drink_can':19,
                  'jar':20,
                  'other':21,
                  'tray':22,
                  'wrap':23},
        'packaging_material': {'cardboard':24,
                  'deformable':25,
                  'glass':26,
                  'metal':27,
                  'other':28,
                  'paper':29,
                  'plastic':30},
        'quantity': {'multiple_products':31,
                  'pack':32,
                  'solo':33},
        'shape': {'circular':34,
                  'conical':35,
                  'curved_rounded':36,
                  'cylindrical':37,
                  'deformable':38,
                  'rectangular':39,
                  'square':40,
                  'trapezoid':41},
        'usage': {'NA':42,
                  'bottom_plug':43,
                  'lid':44,
                  'pressuring_caps':45,
                  'spray':46,
                  'top_plug':47}}
n_attributes = 48

attribute_index2attribute = {}
for attribute_category, dic2 in attribute2attribute_index.items():
    for attribute_value, attribute_index in dic2.items():
        attribute_index2attribute[attribute_index] = attribute_category + " " + attribute_value
