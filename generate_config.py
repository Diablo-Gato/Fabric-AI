import json
import random
from pathlib import Path

random.seed(42)

weathers = ["clear", "rainy", "foggy"]
times = ["morning", "evening", "night", "midday"]
road_types = ["asphalt", "asphalt", "asphalt", "sand"]  # 75% asphalt

configs = []
for i in range(100):
    weather = random.choice(weathers)

    if weather == "clear":
        cow = random.random() < 0.75
        num_autos = random.randint(3, 7)
        density = random.uniform(0.4, 1.0)
    elif weather == "rainy":
        cow = False
        num_autos = random.randint(1, 5)
        density = random.uniform(0.5, 1.0)
    else:  # foggy
        cow = False
        num_autos = random.randint(2, 5)
        density = random.uniform(0.4, 0.9)

    configs.append({
        "weather": weather,
        "time_of_day": random.choice(times),
        "object_density": round(density, 2),
        "auto_rickshaw_count": num_autos,
        "cow": cow,
        "road_type": random.choice(road_types),
        "buildings": random.random() < 0.6
    })

Path("configs").mkdir(exist_ok=True)
with open("configs/scene_configs_100.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Generated {len(configs)} configs → configs/scene_configs_100.json")