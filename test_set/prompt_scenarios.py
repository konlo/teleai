"""Load versioned prompt journeys without importing the agent or connecting a DB."""
from hashlib import sha256
import json
from pathlib import Path

SCENARIO_ROOT = Path(__file__).resolve().parent / 'prompt_scenarios'


def list_scenarios(directory=SCENARIO_ROOT):
    manifest = json.loads((Path(directory) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported scenario manifest version')
    entries = manifest['scenarios']
    names = [entry['name'] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate scenario names')
    return entries


def load_scenario(name, directory=SCENARIO_ROOT):
    directory = Path(directory)
    matches = [entry for entry in list_scenarios(directory) if entry['name'] == name]
    if len(matches) != 1:
        raise ValueError('Unknown scenario: ' + name)
    entry = matches[0]
    filename = entry['file']
    if Path(filename).name != filename or not filename.endswith('.json'):
        raise ValueError('Scenario file must be a JSON file in the scenario directory')
    raw = (directory / filename).read_bytes()
    if sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Scenario checksum mismatch; review and update the manifest')
    data = json.loads(raw)
    turns = data.get('turns', [])
    if (data.get('schema_version') != 1 or data.get('name') != name
            or len(turns) != entry['prompt_count'] or data.get('prompt_count') != len(turns)):
        raise ValueError('Scenario identity/count mismatch')
    if [turn['id'] for turn in turns] != list(range(1, len(turns) + 1)):
        raise ValueError('Scenario turn order must be contiguous')
    if any(not isinstance(t.get('prompt'), str) or not t['prompt'].strip() for t in turns):
        raise ValueError('Every turn needs a nonempty prompt')
    digest = sha256(json.dumps([t['prompt'] for t in turns], ensure_ascii=False,
                              separators=(',', ':')).encode()).hexdigest()
    if digest != data['prompts_sha256']:
        raise ValueError('Prompt text/order mismatch')
    if data.get('execution') != {
        'session': 'new_isolated_conversation', 'turn_order': 'sequential',
        'intent_mode': 'llm', 'automatic_resubmit': False, 'continue_after_incomplete': True,
    }:
        raise ValueError('Unsupported scenario execution contract')
    return data


def prompt_turns(name, directory=SCENARIO_ROOT):
    """Return exact text, including original whitespace, in conversation order."""
    return [turn['prompt'] for turn in load_scenario(name, directory)['turns']]
