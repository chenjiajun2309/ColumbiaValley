import os
import json
import argparse
from datetime import datetime

from modules.maze import Maze
from start import personas

file_markdown = "simulation.md"
file_movement = "movement.json"

frames_per_step = 60  # Number of frames per step


# Read stride from checkpoint files
def get_stride(json_files):
    if len(json_files) < 1:
        return 1

    with open(json_files[-1], "r", encoding="utf-8") as f:
        config = json.load(f)

    return config["stride"]


# Convert address to string
def get_location(address):
    # For compatibility with original version only
    # if address[0] == "<waiting>" or address[0] == "<persona>":
    #     return None

    # Don't need to display the first level of address (e.g., "the Ville")
    location = "，".join(address[1:])

    return location


# Insert frame 0 data (Agent's initial state)
def insert_frame0(init_pos, movement, agent_name):
    key = "0"
    if key not in movement.keys():
        movement[key] = dict()

    json_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        address = json_data["spatial"]["address"]["living_area"]
    location = get_location(address)
    coord = json_data["coord"]
    init_pos[agent_name] = coord
    movement[key][agent_name] = {
        "location": location,
        "movement": coord,
        "description": "sleeping",
    }
    movement["description"][agent_name] = {
        "currently": json_data["currently"],
        "scratch": json_data["scratch"],
    }


# Extract data from all checkpoint files (for replay)
def generate_movement(checkpoints_folder, compressed_folder, compressed_file):
    movement_file = os.path.join(compressed_folder, compressed_file)

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    files = sorted(os.listdir(checkpoints_folder))
    json_files = list()
    for file_name in files:
        if file_name.endswith(".json") and file_name != conversation_file:
            json_files.append(os.path.join(checkpoints_folder, file_name))

    persona_init_pos = dict()
    all_movement = dict()
    all_movement["description"] = dict()
    all_movement["conversation"] = dict()

    stride = get_stride(json_files)
    sec_per_step = stride

    result = {
        "start_datetime": "",  # Start time
        "stride": stride,  # Minutes per step (must match generation parameters)
        "sec_per_step": sec_per_step,  # Seconds per frame during replay
        "persona_init_pos": persona_init_pos,  # Initial position of each Agent
        "all_movement": all_movement,  # Position changes of all Agents in each step
    }

    last_location = dict()

    # Load map data for calculating Agent movement paths
    json_path = "frontend/static/assets/village/maze.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        maze = Maze(json_data, None)

    for file_name in json_files:
        # Read all checkpoint files sequentially
        with open(file_name, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            step = json_data["step"]
            agents = json_data["agents"]

            # Save replay start time
            if len(result["start_datetime"]) < 1:
                t = datetime.strptime(json_data["time"], "%Y%m%d-%H:%M")
                result["start_datetime"] = t.isoformat()

            # Iterate through all Agents in a single checkpoint file
            for agent_name, agent_data in agents.items():
                # Insert frame 0
                if step == 1:
                    insert_frame0(persona_init_pos, all_movement, agent_name)

                source_coord = last_location.get(agent_name, all_movement["0"][agent_name])["movement"]
                target_coord = agent_data["coord"]
                location = get_location(agent_data["action"]["event"]["address"])
                if location is None:
                    location = last_location.get(agent_name, all_movement["0"][agent_name])["location"]
                    path = [source_coord]
                else:
                    path = maze.find_path(source_coord, target_coord)

                had_conversation = False
                step_conversation = ""
                persons_in_conversation = []
                step_time = json_data["time"]
                if step_time in conversation.keys():
                    for chats in conversation[step_time]:
                        for persons, chat in chats.items():
                            persons_in_conversation.append(persons.split(" @ ")[0].split(" -> "))
                            step_conversation += f"\nLocation: {persons.split(' @ ')[1]}\n\n"
                            for c in chat:
                                agent = c[0]
                                text = c[1]
                                step_conversation += f"{agent}：{text}\n"

                for i in range(frames_per_step):
                    moving = len(path) > 1
                    if len(path) > 0:
                        movement = list(path[0])
                        path = path[1:]
                        if agent_name not in last_location.keys():
                            last_location[agent_name] = dict()
                        last_location[agent_name]["movement"] = movement
                        last_location[agent_name]["location"] = location
                    else:
                        movement = None

                    if moving:
                        action = f"Going to {location}"
                    elif movement is not None:
                        action = agent_data["action"]["event"]["describe"]
                        if len(action) < 1:
                            action = f'{agent_data["action"]["event"]["predicate"]}{agent_data["action"]["event"]["object"]}'

                        # Check if current Agent has new conversation in this checkpoint file (for setting icon)
                        for persons in persons_in_conversation:
                            if agent_name in persons:
                                had_conversation = True
                                break

                        # Set icons for sleeping and conversation
                        if "sleeping" in action or "sleep" in action:
                            action = "😴 " + action
                        elif had_conversation:
                            action = "💬 " + action

                    step_key = "%d" % ((step-1) * frames_per_step + 1 + i)
                    if step_key not in all_movement.keys():
                        all_movement[step_key] = dict()

                    if movement is not None:
                        all_movement[step_key][agent_name] = {
                            "location": location,
                            "movement": movement,
                            "action": action,
                        }
                all_movement["conversation"][step_time] = step_conversation

    # Save data
    with open(movement_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# Generate Markdown document
def generate_report(checkpoints_folder, compressed_folder, compressed_file):
    last_state = dict()

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    def extract_description():
        markdown_content = "# Basic Persona\n\n"
        for agent_name in personas:
            json_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                markdown_content += f"## {agent_name}\n\n"
                markdown_content += f"Age: {json_data['scratch']['age']} years old  \n"
                markdown_content += f"Innate: {json_data['scratch']['innate']}  \n"
                markdown_content += f"Learned: {json_data['scratch']['learned']}  \n"
                markdown_content += f"Lifestyle: {json_data['scratch']['lifestyle']}  \n"
                markdown_content += f"Current state: {json_data['currently']}\n\n"
        return markdown_content

    def extract_action(json_data):
        markdown_content = ""
        agents = json_data["agents"]
        for agent_name, agent_data in agents.items():
            if agent_name not in last_state.keys():
                last_state[agent_name] = {"currently": "", "location": "", "action": ""}

            location = "，".join(agent_data["action"]["event"]["address"])
            action = agent_data["action"]["event"]["describe"]

            if location == last_state[agent_name]["location"] and action == last_state[agent_name]["action"]:
                continue

            last_state[agent_name]["location"] = location
            last_state[agent_name]["action"] = action

            if len(markdown_content) < 1:
                markdown_content = f"# {json_data['time']}\n\n"
                markdown_content += "## Activity Log:\n\n"

            markdown_content += f"### {agent_name}\n"

            if len(action) < 1:
                action = "sleeping"

            markdown_content += f"Location: {location}  \n"
            markdown_content += f"Activity: {action}  \n"

            markdown_content += f"\n"

        if json_data['time'] not in conversation.keys():
            return markdown_content

        markdown_content += "## Conversation Log:\n\n"
        for chats in conversation[json_data['time']]:
            for agents, chat in chats.items():
                markdown_content += f"### {agents}\n\n"
                for item in chat:
                    markdown_content += f"`{item[0]}`\n> {item[1]}\n\n"
        return markdown_content

    all_markdown_content = extract_description()
    files = sorted(os.listdir(checkpoints_folder))
    for file_name in files:
        if (not file_name.endswith(".json")) or (file_name == conversation_file):
            continue

        file_path = os.path.join(checkpoints_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            content = extract_action(json_data)
            all_markdown_content += content + "\n\n"
    with open(f"{compressed_folder}/{compressed_file}", "w", encoding="utf-8") as compressed_file:
        compressed_file.write(all_markdown_content)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="the name of the simulation")
args = parser.parse_args()


if __name__ == "__main__":
    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name: ")

    while not os.path.exists(f"results/checkpoints/{name}"):
        name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")

    checkpoints_folder = f"results/checkpoints/{name}"
    compressed_folder = f"results/compressed/{name}"
    os.makedirs(compressed_folder, exist_ok=True)

    generate_report(checkpoints_folder, compressed_folder, file_markdown)
    generate_movement(checkpoints_folder, compressed_folder, file_movement)
