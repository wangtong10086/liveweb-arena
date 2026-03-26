import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.think_ablation.common import (
    crop_observation_prompt,
    exact_action_match,
    normalize_action,
    normalized_action_match,
    observation_overlap_ratio,
)


def test_normalize_action_denormalizes_stop():
    action = {"action_type": "stop", "params": {"final": {"answers": {"answer1": "42"}}}}
    normalized = normalize_action(action)
    assert normalized == {"action_type": "stop", "params": {"answers": {"answer1": "42"}}}


def test_exact_and_normalized_action_match_for_goto():
    reference = {"action_type": "goto", "params": {"url": "https://news.ycombinator.com/ask/"}}
    predicted = {"action_type": "goto", "params": {"url": "http://news.ycombinator.com/ask"}}
    assert not exact_action_match(reference, predicted)
    assert normalized_action_match(reference, predicted)


def test_crop_observation_prompt_only_crops_tree():
    prompt = (
        "### Accessibility Tree\n```\n"
        + "A" * 200
        + "\n```\n\nWhat is your next action?"
    )
    cropped = crop_observation_prompt(prompt, 50)
    assert len(cropped) < len(prompt)
    assert "What is your next action?" in cropped


def test_observation_overlap_ratio_uses_page_tokens():
    think = "I should click the ask hn link on hacker news to compare the ask posts."
    observation = {
        "url": "https://news.ycombinator.com/",
        "title": "Hacker News",
        "accessibility_tree": "link Ask HN\nlink Show HN\nlink Jobs",
    }
    ratio = observation_overlap_ratio(think, observation)
    assert ratio > 0.2
