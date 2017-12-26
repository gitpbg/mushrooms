import pandas as pd


# Attribute Information: (classes: edible=e, poisonous=p)
#     cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
#     cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
#     cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
#     bruises: bruises=t,no=f
#     odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
#     gill-attachment: attached=a,descending=d,free=f,notched=n

#     gill-spacing: close=c,crowded=w,distant=d
#     gill-size: broad=b,narrow=n
#     gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
#     stalk-shape: enlarging=e,tapering=t
#     stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
#     stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#     stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#     stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#     stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#     veil-type: partial=p,universal=u
#     veil-color: brown=n,orange=o,white=w,yellow=y
#     ring-number: none=n,one=o,two=t
#     ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
#     spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
#     population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
#     habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d


dataset = [
  ('class', [('e', 'edible'), ('p', 'poisonous')]),
  ('cap-shape', [
    ('b', 'bell'),
    ('c', 'conical'),
    ('x', 'convex'),
    ('f', 'flat'),
    ('k', 'knobbed'),
    ('s', 'sunken')
  ]),
  ('cap-surface',[
    ('f', 'fibrous'),
    ('g', 'grooves'),
    ('y', 'scaly'),
    ('s', 'smooth')
  ]),
#     cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
  ('cap-color', [
    ('n', 'brown'),
    ('b', 'buff'),
    ('c', 'cinnamon'),
    ('g', 'gray'),
    ('r', 'green'),
    ('p', 'pink'),
    ('u', 'purple'),
    ('e', 'red'),
    ('w', 'white'),
    ('y', 'yellow')
  ]),
#     bruises: bruises=t,no=f
  ('bruises', [
    ('t', 'bruised'),
    ('f', 'notbruised')
  ]),
#     odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
  ('odor', [
    ('a', 'almond'),
    ('l', 'anise'),
    ('c', 'creosote'),
    ('y', 'fishy'),
    ('f', 'foul'),
    ('m', 'musty'),
    ('n', 'none'),
    ('p', 'pungent'),
    ('s', 'spicy')
  ]),
#     gill-attachment: attached=a,descending=d,free=f,notched=n
  ('gill-attachment', [
    ('a', 'attached'),
    ('d', 'descending'),
    ('f', 'free'),
    ('n', 'notched')
  ]),
#     gill-spacing: close=c,crowded=w,distant=d
  ('gill-spacing', [
    ('c', 'close'),
    ('w', 'crowded'),
    ('d', 'distant')
  ]),
#     gill-size: broad=b,narrow=n
  ('gill-size', [
    ('b', 'broad'),
    ('n', 'narrow')
  ]),
#     gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
  ('gill-color', [
    ('k', 'black'),
    ('n', 'brown'),
    ('b', 'buff'),
    ('h', 'chocolate'),
    ('g', 'gray'),
    ('r', 'green'),
    ('o', 'orange'),
    ('p', 'pink'),
    ('u', 'purple'),
    ('e', 'red'),
    ('w', 'white'),
    ('y', 'yellow')
  ]),
#     stalk-shape: enlarging=e,tapering=t
  ('stalk-shape',[
    ('e', 'enlarging'),
    ('t', 'tapering')
  ]),
#     stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
  ('stalk-root', [
    ('b', 'bulbous'),
    ('c', 'club'),
    ('u', 'cup'),
    ('e', 'equal'),
    ('z', 'rhizomorphs'),
    ('r', 'rooted'),
    ('?', 'missing')
  ]),
#     stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
  ('stalk-surface-above-ring',[
    ('f', 'fibrous'),
    ('y', 'scaly'),
    ('k', 'silky'),
    ('s', 'smooth')
  ]),
#     stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
  ('stalk-surface-below-ring',[
    ('f', 'fibrous'),
    ('y', 'scaly'),
    ('k', 'silky'),
    ('s', 'smooth')
  ]),
#     stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
  ('stalk-color-above-ring', [
    ('n', 'brown'),
    ('b', 'buff'),
    ('c', 'cinnamon'),
    ('g', 'gray'),
    ('o', 'orange'),
    ('p', 'pink'),
    ('e', 'red'),
    ('w', 'white'),
    ('y', 'yellow')
  ]),
#     stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
  ('stalk-color-below-ring', [
    ('n', 'brown'),
    ('b', 'buff'),
    ('c', 'cinnamon'),
    ('g', 'gray'),
    ('o', 'orange'),
    ('p', 'pink'),
    ('e', 'red'),
    ('w', 'white'),
    ('y', 'yellow')
  ]),
#     veil-type: partial=p,universal=u
  ('veil-type', [
    ('p', 'partial'),
    ('u', 'universal')
  ]),
#     veil-color: brown=n,orange=o,white=w,yellow=y
  ('veil-color', [
    ('n', 'brown'),
    ('o', 'orange'),
    ('w', 'white'),
    ('y', 'yellow')
  ]),
#     ring-number: none=n,one=o,two=t
  ('ring-number', [
    ('n', 'none'),
    ('o', 'one'),
    ('t', 'two')
  ]),
#     ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
  ('ring-type', [
    ('c', 'cobwebby'),
    ('e', 'evanescent'),
    ('f', 'flaring'),
    ('l', 'large'),
    ('n', 'none'),
    ('p', 'pendant'),
    ('s', 'sheathing'),
    ('z', 'zone')
  ]),
#     spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
  ('spore-print-color', [
    ('k', 'black'),
    ('n', 'brown'),
    ('b', 'buff'),
    ('h', 'chocolate'),
    ('r', 'green'),
    ('o', 'orange'),
    ('p', 'pink'),
    ('u', 'purple'),
    ('e', 'red'),
    ('w', 'white'),
    ('y', 'yellow')
    
  ]),
#     population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
  ('population', [
    ('a', 'abundant'),
    ('c', 'clustered'),
    ('n', 'numerous'),
    ('s', 'scattered'),
    ('v', 'several'),
    ('y', 'solitary')
  ]),
#     habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
  ('habitat', [
    ('g', 'grasses'),
    ('l', 'leaves'),
    ('m', 'meadows'),
    ('p', 'paths'),
    ('u', 'urban'),
    ('w', 'waste'),
    ('d', 'woods')
  ])
]

class MyDict(dict):
    def __init__(self):
      super(dict)
    
    def __missing__(self, v):
      return 0

def make_dataframe():
  frame = pd.read_csv("mushrooms.csv")
  print(frame.head())
  md = MyDict()
  for (column, expansion) in dataset:
    print("Column: " + column)
    for (ch, colname) in expansion:
      md.clear()
      md[ch] = 1
      params = {}
      params[column+"_" +colname] = lambda x: x[column].map(md)
      #print("Adding Column " + colname)
      frame = frame.assign(**params)
    del(frame[column])

  return frame