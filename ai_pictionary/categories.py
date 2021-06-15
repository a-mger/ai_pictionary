
labels = [
    'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'apple', 'arm', 'asparagus', 'axe',
    'backpack', 'banana', 'barn', 'baseball bat',
    'basket', 'basketball', 'bat', 'bathtub', 'bear', 'beard', 'bed',
    'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake',
    'blackberry', 'book', 'boomerang', 'bowtie', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
    'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator',
    'camel', 'camera', 'campfire', 'candle',
    'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat',
    'cello', 'cell phone', 'chair', 'chandelier', 'church', 'circle',
    'clock', 'cloud', 'compass', 'computer',
    'cookie', 'cooler', 'couch', 'cow', 'crab', 'crocodile', 'crown',
    'cup', 'diamond', 'dog',
    'dolphin', 'donut', 'door', 'dragon', 'drill', 'drums', 'duck',
    'dumbbell', 'ear', 'elephant', 'envelope', 'eraser', 'eye',
    'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger',
    'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops',
    'flower', 'flying saucer', 'foot', 'fork', 'frog',
    'giraffe', 'goatee',
    'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
    'headphones', 'hedgehog', 'helicopter',
    'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog',
    'hourglass', 'house', 'hurricane', 'ice cream',
    'jacket', 'kangaroo', 'key', 'keyboard', 'knife', 'ladder',
    'lantern', 'laptop', 'leaf', 'light bulb', 'lighthouse',
    'lightning', 'lion', 'lobster', 'lollipop', 'mailbox',
    'map', 'matches', 'megaphone', 'mermaid', 'microphone',
    'monkey', 'moon', 'mosquito', 'motorbike', 'mountain',
    'mouse', 'moustache', 'mouth', 'mushroom',
    'nose', 'ocean', 'octopus', 'onion', 'oven', 'owl',
    'paintbrush', 'paint can', 'palm tree', 'panda', 'pants', 'paper clip',
    'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil',
    'penguin', 'piano', 'pickup truck', 'picture frame', 'pig',
    'pineapple', 'pizza', 'police car',
    'postcard', 'potato', 'purse', 'rabbit', 'raccoon',
    'radio', 'rainbow', 'rake', 'remote control', 'rhinoceros',
    'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich',
    'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver',
    'sea turtle', 'shark', 'sheep', 'shoe', 'shorts', 'shovel',
    'skateboard', 'skull', 'skyscraper', 'smiley face',
    'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock',
    'speedboat', 'spider', 'spoon', 'square',
    'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stitches',
    'stop sign', 'strawberry', 'streetlight', 'string bean',
    'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword',
    'syringe', 'table', 'teapot', 'teddy-bear', 'telephone', 'television',
    'tennis racquet', 'tent', 'The Eiffel Tower',
    'The Mona Lisa', 'tiger', 'toaster', 'toilet', 'tooth',
    'toothbrush', 'toothpaste', 'tornado', 'traffic light', 'train',
    'tree', 'truck', 'trumpet', 't-shirt', 'umbrella',
    'van', 'vase', 'violin', 'washing machine', 'watermelon',
    'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass',
    'wristwatch', 'yoga', 'zebra'
]

labels_250 = ['oven', 'megaphone', 'camera', 'shark', 'raccoon', 'traffic light',
       'apple', 'snowman', 'stereo', 'crown', 'alarm clock', 'camel',
       'chair', 'door', 'panda', 'penguin', 'rainbow', 'basketball',
       'cannon', 'carrot', 'goatee', 'bicycle', 'postcard', 'angel',
       'screwdriver', 'blackberry', 'snorkel', 'saw', 'tornado',
       'skateboard', 'flamingo', 'owl', 'cow', 'snowflake', 'dumbbell',
       'speedboat', 'campfire', 'sailboat', 'keyboard', 'snake',
       'hourglass', 'clock', 'string bean', 'moustache', 'house',
       'streetlight', 'cactus', 'watermelon', 'beard', 'key', 'yoga',
       'crocodile', 'washing machine', 'train', 'soccer ball', 'bed',
       'parachute', 'sea turtle', 'airplane', 'telephone', 'motorbike',
       'headphones', 'shovel', 'diamond', 'palm tree', 'donut',
       'remote control', 'kangaroo', 'dragon', 'mountain', 'grass',
       'barn', 'horse', 'pickup truck', 'hot dog', 'purse', 'cloud',
       'hockey stick', 'canoe', 'drill', 'ambulance', 'potato', 'bush',
       'bathtub', 'ladder', 'The Mona Lisa', 'peanut', 'matches',
       'compass', 'laptop', 'waterslide', 'jacket', 'teddy-bear',
       'picture frame', 'sword', 'sun', 'foot', 'calculator', 'car',
       'tree', 'church', 'leaf', 'belt', 'butterfly', 'star', 'pencil',
       'banana', 'skull', 'hand', 'tooth', 'umbrella', 'asparagus', 'van',
       'pineapple', 'bridge', 'bear', 'monkey', 'eye', 'stitches',
       'sandwich', 'truck', 'snail', 'syringe', 'brain', 'wine glass',
       'spoon', 'toilet', 'eraser', 'pants', 'teapot', 'ear', 'submarine',
       'television', 'light bulb', 'rollerskates', 'rake', 'cello',
       'baseball bat', 'whale', 'trumpet', 'bowtie', 'rifle', 'bread',
       'skyscraper', 'lantern', 'river', 'squirrel', 'hedgehog', 'parrot',
       'computer', 'wine bottle', 'shoe', 'sheep', 'flying saucer',
       'flashlight', 'microphone', 'zebra', 'rhinoceros', 'knife',
       'hamburger', 'feather', 'bench', 'finger', 'dog', 'mermaid',
       'pizza', 'bucket', 'axe', 'envelope', 'police car', 'lightning',
       'toothbrush', 't-shirt', 'vase', 'giraffe', 'roller coaster',
       'swing set', 'hot air balloon', 'wristwatch', 'square',
       'toothpaste', 'moon', 'scissors', 'ocean', 'smiley face',
       'mailbox', 'bat', 'suitcase', 'mouse', 'paint can', 'shorts',
       'grapes', 'toaster', 'mouth', 'bus', 'octopus', 'paper clip',
       'mushroom', 'cake', 'radio', 'hat', 'duck', 'cookie', 'paintbrush',
       'castle', 'basket', 'chandelier', 'backpack', 'guitar', 'rabbit',
       'book', 'strawberry', 'ant', 'circle', 'table', 'binoculars',
       'helicopter', 'face', 'fork', 'flower', 'fish', 'tennis racquet',
       'lion', 'candle', 'peas', 'crab', 'fan', 'hurricane', 'arm',
       'dolphin', 'cat', 'nose', 'school bus', 'bee', 'The Eiffel Tower',
       'frog', 'fireplace', 'boomerang', 'steak', 'eyeglasses', 'hammer',
       'violin', 'sweater', 'firetruck', 'couch', 'saxophone', 'sock',
       'lollipop', 'map', 'elephant']

labels_150 = ['teapot', 'violin', 'hammer', 'basketball', 'hedgehog', 'bridge',
       'tree', 'swing set', 'car', 'speedboat', 'canoe', 'mouth',
       'feather', 'alarm clock', 'barn', 'strawberry', 'megaphone',
       'truck', 'grass', 'radio', 'headphones', 'fish', 'shorts',
       'television', 'dumbbell', 'bench', 'fork', 'giraffe', 'lightning',
       'computer', 'table', 'monkey', 'hot dog', 'waterslide',
       'police car', 'panda', 'bread', 'roller coaster', 'firetruck',
       'calculator', 'envelope', 'tooth', 'penguin', 'cloud', 'angel',
       'moon', 'dolphin', 'diamond', 'ambulance', 'sweater', 'eraser',
       'circle', 'toothpaste', 'ant', 'whale', 'pickup truck', 'bear',
       'square', 'screwdriver', 'umbrella', 'bathtub', 'flower', 'rabbit',
       'wine bottle', 'submarine', 'potato', 'crab', 'sheep', 'bat',
       't-shirt', 'zebra', 'backpack', 'sailboat', 'sandwich', 'star',
       'snake', 'mountain', 'door', 'snowflake', 'ocean', 'snowman',
       'sea turtle', 'chair', 'snorkel', 'motorbike', 'eyeglasses',
       'camera', 'drill', 'saxophone', 'hockey stick', 'banana',
       'chandelier', 'cat', 'purse', 'saw', 'soccer ball', 'bus', 'house',
       'flashlight', 'peas', 'laptop', 'octopus', 'microphone', 'pizza',
       'donut', 'snail', 'bowtie', 'belt', 'basket', 'dog', 'axe',
       'crown', 'campfire', 'cello', 'butterfly', 'key', 'mushroom',
       'wristwatch', 'vase', 'skull', 'mouse', 'paper clip', 'teddy-bear',
       'hot air balloon', 'keyboard', 'boomerang', 'oven', 'light bulb',
       'hamburger', 'church', 'candle', 'crocodile', 'brain',
       'skateboard', 'bush', 'suitcase', 'picture frame', 'hourglass',
       'smiley face', 'castle', 'steak', 'asparagus', 'compass',
       'trumpet', 'lollipop', 'arm', 'mermaid', 'airplane', 'duck', 'van']

labels_50 = ['key', 'hot air balloon', 'suitcase', 'hedgehog', 'angel',
       'basketball', 'sandwich', 'candle', 'motorbike', 'crocodile',
       'laptop', 'cloud', 'toothpaste', 'bowtie', 'sailboat', 't-shirt',
       'snowflake', 'flashlight', 'belt', 'bush', 'steak', 'butterfly',
       'computer', 'star', 'banana', 'house', 'crown', 'penguin',
       'monkey', 'campfire', 'mouse', 'fork', 'bus', 'chandelier', 'duck',
       'eraser', 'van', 'cat', 'zebra', 'saw', 'picture frame', 'flower',
       'envelope', 'alarm clock', 'saxophone', 'sweater', 'calculator',
       'hockey stick', 'submarine', 'hamburger']