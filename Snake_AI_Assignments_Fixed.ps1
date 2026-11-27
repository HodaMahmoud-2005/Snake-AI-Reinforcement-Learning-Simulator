param([switch]$Preview)
$ErrorActionPreference = 'Stop'
$repo = 'HodaMahmoud-2005/Snake-AI-Reinforcement-Learning-Simulator'
$issues = @'
[
  {
    "id": 1,
    "title": "Initialize project structure and dependencies",
    "owner": "Hany",
    "assignee": "hanyzead123",
    "labels": [
      "environment",
      "priority: high"
    ],
    "body": "## Objective\nEstablish the shared Python workspace so every member can work independently.\n\n## Ownership and scope\n- Team member: Hany (@hanyzead123)\n- Backlog ID: SNAKE-01\n- Priority: High\n- Files: requirements.txt, .gitignore, models/, results/, tests/, report/\n- Dependencies (backlog references, not GitHub issue numbers): None\n\n## Tasks and acceptance criteria\n- [ ] Create all agreed modules and folders; use .gitkeep where an otherwise empty directory must be tracked.\n- [ ] List NumPy, Pygame, Matplotlib and Pandas in requirements.txt; include pytest for testing.\n- [ ] Ignore virtual environments, Python caches and temporary files without ignoring required final deliverables.\n- [ ] Verify a clean installation using python -m pip install -r requirements.txt.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 2,
    "title": "Define game configuration and hyperparameters",
    "owner": "Hany",
    "assignee": "hanyzead123",
    "labels": [
      "environment",
      "priority: high"
    ],
    "body": "## Objective\nCentralize game rules and baseline hyperparameters to ensure comparable experiments.\n\n## Ownership and scope\n- Team member: Hany (@hanyzead123)\n- Backlog ID: SNAKE-02\n- Priority: High\n- Files: config.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 01: Initialize project structure and dependencies\n\n## Tasks and acceptance criteria\n- [ ] Set a 5x5 grid, initial snake length 2, three apples to win, two fixed obstacles and 60-step limit.\n- [ ] Define actions 0=left, 1=straight, 2=right relative to the snake heading.\n- [ ] Set food +10, extra win bonus +100, ordinary move -1 and collision -20; final apple reward is +110.\n- [ ] Expose gamma=0.95, Q-learning alpha=0.1, epsilon 1.0 to 0.05 with decay 0.9995, REINFORCE alpha=0.01, 10,000 training episodes and Value Iteration tolerance 1e-5 as configurable starting values.\n- [ ] Document approved start states, fixed food sequence and deterministic fallback for occupied food cells; keep random seeds configurable.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 3,
    "title": "Implement core Snake environment",
    "owner": "Hany",
    "assignee": "hanyzead123",
    "labels": [
      "environment",
      "priority: high"
    ],
    "body": "## Objective\nImplement the shared game independently of rendering or any learning algorithm.\n\n## Ownership and scope\n- Team member: Hany (@hanyzead123)\n- Backlog ID: SNAKE-03\n- Priority: High\n- Files: environment.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 02: Define game configuration and hyperparameters\n\n## Tasks and acceptance criteria\n- [ ] Implement reset(), get_state() and step(action); return next state, reward, terminated, truncated and info.\n- [ ] Implement three relative actions, legal movement, body growth and deterministic food spawning.\n- [ ] Detect wall, obstacle and self collisions; allow movement into the old tail cell when it vacates on a non-growth move.\n- [ ] Apply shared reward rules, win after three apples and truncate at 60 steps without adding an unstated penalty.\n- [ ] Expose score, steps and termination reason for evaluation; do not render during training.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 4,
    "title": "Implement deterministic transition interface",
    "owner": "Hany",
    "assignee": "hanyzead123",
    "labels": [
      "environment",
      "dynamic-programming",
      "priority: high"
    ],
    "body": "## Objective\nProvide a pure model of the same game for Dynamic Programming.\n\n## Ownership and scope\n- Team member: Hany (@hanyzead123)\n- Backlog ID: SNAKE-04\n- Priority: High\n- Files: environment.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 03: Implement core Snake environment\n\n## Tasks and acceptance criteria\n- [ ] Represent state as an immutable ordered snake-body tuple plus food position; derive heading and apples from body.\n- [ ] Implement transition(state, action) returning next state, reward and true-terminal flag without mutating a live game.\n- [ ] Use deterministic food placement and a consistent collision rule shared by step() and transition().\n- [ ] Keep elapsed steps outside the planning state as an external truncation limit; document that planning solves the uncapped discounted MDP.\n- [ ] Verify that identical states and actions yield identical outcomes.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 5,
    "title": "Test and document the Snake environment",
    "owner": "Hany",
    "assignee": "hanyzead123",
    "labels": [
      "testing",
      "documentation",
      "priority: high"
    ],
    "body": "## Objective\nProve the shared environment is correct before training the agents.\n\n## Ownership and scope\n- Team member: Hany (@hanyzead123)\n- Backlog ID: SNAKE-05\n- Priority: High\n- Files: tests/test_environment.py, report/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 03: Implement core Snake environment, Backlog 04: Implement deterministic transition interface\n\n## Tasks and acceptance criteria\n- [ ] Test reset, relative turns, normal movement, growth, food placement and tail-cell movement.\n- [ ] Test wall, obstacle and body collision, all reward values, three-apple win and 60-step truncation.\n- [ ] Test step/transition consistency for non-truncated transitions and verify transition does not mutate state.\n- [ ] Create a labeled grid/map, state-transition illustration and environment flowchart or pseudocode.\n- [ ] Add environment rules and test evidence to the report.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 6,
    "title": "Enumerate reachable Snake states using BFS",
    "owner": "Hoda",
    "assignee": "HodaMahmoud-2005",
    "labels": [
      "dynamic-programming",
      "priority: high"
    ],
    "body": "## Objective\nDiscover only states that can actually occur from the approved initial-state set.\n\n## Ownership and scope\n- Team member: Hoda (@HodaMahmoud-2005)\n- Backlog ID: SNAKE-06\n- Priority: High\n- Files: value_iteration.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 04: Implement deterministic transition interface\n\n## Tasks and acceptance criteria\n- [ ] Use a BFS queue and visited set to enumerate full states from all permitted starting configurations.\n- [ ] Expand all three actions using transition(); do not expand true terminal outcomes.\n- [ ] Deduplicate states and record the number of reachable nonterminal states.\n- [ ] Report enumeration runtime and handle resource limits explicitly rather than silently omitting states.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 7,
    "title": "Build the environment transition model",
    "owner": "Hoda",
    "assignee": "HodaMahmoud-2005",
    "labels": [
      "dynamic-programming",
      "priority: high"
    ],
    "body": "## Objective\nCache the model required by Bellman updates.\n\n## Ownership and scope\n- Team member: Hoda (@HodaMahmoud-2005)\n- Backlog ID: SNAKE-07\n- Priority: High\n- Files: value_iteration.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 06: Enumerate reachable Snake states using BFS\n\n## Tasks and acceptance criteria\n- [ ] Build a state-action mapping with next state, reward and terminal flag for every reachable nonterminal state and all three actions.\n- [ ] Reuse the shared transition implementation rather than copying game rules.\n- [ ] Validate that nonterminal successor states exist in the enumerated set and that terminal outcomes require no bootstrap.\n- [ ] Record model size and construction time.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 8,
    "title": "Implement Value Iteration from scratch",
    "owner": "Hoda",
    "assignee": "HodaMahmoud-2005",
    "labels": [
      "dynamic-programming",
      "priority: high"
    ],
    "body": "## Objective\nCompute discounted state values with the Bellman optimality equation.\n\n## Ownership and scope\n- Team member: Hoda (@HodaMahmoud-2005)\n- Backlog ID: SNAKE-08\n- Priority: High\n- Files: value_iteration.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 07: Build the environment transition model\n\n## Tasks and acceptance criteria\n- [ ] Initialize V(s)=0 and use shared gamma.\n- [ ] Implement synchronous Bellman sweeps with max over three actions; use zero future value for terminal transitions.\n- [ ] Track max absolute value change per sweep; stop when it falls below tolerance or an explicit iteration cap.\n- [ ] Report whether convergence was achieved and avoid prebuilt RL algorithm implementations.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 9,
    "title": "Extract optimal policy and record convergence",
    "owner": "Hoda",
    "assignee": "HodaMahmoud-2005",
    "labels": [
      "dynamic-programming",
      "priority: high"
    ],
    "body": "## Objective\nTurn converged state values into a usable policy and planning metrics.\n\n## Ownership and scope\n- Team member: Hoda (@HodaMahmoud-2005)\n- Backlog ID: SNAKE-09\n- Priority: High\n- Files: value_iteration.py, models/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 08: Implement Value Iteration from scratch\n\n## Tasks and acceptance criteria\n- [ ] Extract argmax action for each reachable state using reward plus discounted successor value.\n- [ ] Use documented deterministic tie-breaking and handle unseen states explicitly.\n- [ ] Export policy, state count, residual history, convergence status and planning time through the shared model format.\n- [ ] Include state enumeration and model construction in total planning time.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 10,
    "title": "Test and document Value Iteration",
    "owner": "Hoda",
    "assignee": "HodaMahmoud-2005",
    "labels": [
      "testing",
      "documentation",
      "dynamic-programming",
      "priority: medium"
    ],
    "body": "## Objective\nValidate the DP implementation and explain its assumptions.\n\n## Ownership and scope\n- Team member: Hoda (@HodaMahmoud-2005)\n- Backlog ID: SNAKE-10\n- Priority: Medium\n- Files: tests/test_value_iteration.py, report/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 08: Implement Value Iteration from scratch, Backlog 09: Extract optimal policy and record convergence\n\n## Tasks and acceptance criteria\n- [ ] Test a hand-calculated Bellman update and a terminal transition.\n- [ ] Check state coverage, policy validity, residual threshold and successful execution on small known scenarios.\n- [ ] Document the equation, deterministic model, reachable state count and computational complexity.\n- [ ] Explain the difference between planning iterations and training episodes and the external time limit.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 11,
    "title": "Implement Q-Learning agent from scratch",
    "owner": "Basmala",
    "assignee": "Basmala-ElKady",
    "labels": [
      "q-learning",
      "priority: high"
    ],
    "body": "## Objective\nLearn action values from sampled interaction without accessing the transition model.\n\n## Ownership and scope\n- Team member: Basmala (@Basmala-ElKady)\n- Backlog ID: SNAKE-11\n- Priority: High\n- Files: q_learning.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 03: Implement core Snake environment, Backlog 04: Implement deterministic transition interface\n\n## Tasks and acceptance criteria\n- [ ] Use a dictionary-based Q-table keyed by full state with three action values.\n- [ ] Implement the Q-learning TD update using configurable alpha and gamma.\n- [ ] Bootstrap nonterminal transitions, including external truncation; use zero bootstrap for true terminal transitions.\n- [ ] Provide greedy action selection, documented tie-breaking and safe initialization of unseen states.\n- [ ] Do not use a prebuilt RL algorithm.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 12,
    "title": "Implement epsilon-greedy exploration",
    "owner": "Jowairya",
    "assignee": "jowairyakassem",
    "labels": [
      "q-learning",
      "priority: high"
    ],
    "body": "## Objective\nBalance exploration and exploitation during learning.\n\n## Ownership and scope\n- Team member: Jowairya (@jowairyakassem)\n- Backlog ID: SNAKE-12\n- Priority: High\n- Files: q_learning.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 11: Implement Q-Learning agent from scratch\n\n## Tasks and acceptance criteria\n- [ ] Implement uniform random action selection with probability epsilon and greedy selection otherwise.\n- [ ] Decay epsilon per episode from 1.0 toward a configurable floor of 0.05.\n- [ ] Use an explicit seeded random generator and shared configuration.\n- [ ] Disable exploration for greedy evaluation without mutating trained Q-values.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 13,
    "title": "Develop the unified agent training pipeline",
    "owner": "Basmala",
    "assignee": "Basmala-ElKady",
    "labels": [
      "q-learning",
      "integration",
      "priority: high"
    ],
    "body": "## Objective\nProvide consistent orchestration and logging for all three methods.\n\n## Ownership and scope\n- Team member: Basmala (@Basmala-ElKady)\n- Backlog ID: SNAKE-13\n- Priority: High\n- Files: train.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 03: Implement core Snake environment, Backlog 08: Implement Value Iteration from scratch, Backlog 12: Implement epsilon-greedy exploration, Backlog 17: Implement REINFORCE Policy Gradient updates\n\n## Tasks and acceptance criteria\n- [ ] Support selection of Value Iteration, Q-Learning or REINFORCE via a common entry point.\n- [ ] Run episode training for Q-Learning and REINFORCE and iteration-based planning for Value Iteration.\n- [ ] Use comparable initial-state schedules and fixed seeds; keep rendering disabled during timed runs.\n- [ ] Record episode return, steps, apples, success, truncation and elapsed training time; log DP residuals separately.\n- [ ] Expose configurable training budgets and return trained policy plus metrics.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 14,
    "title": "Implement model saving and loading",
    "owner": "Basmala",
    "assignee": "Basmala-ElKady",
    "labels": [
      "integration",
      "priority: medium"
    ],
    "body": "## Objective\nMake trained policies reusable in the demo and evaluation.\n\n## Ownership and scope\n- Team member: Basmala (@Basmala-ElKady)\n- Backlog ID: SNAKE-14\n- Priority: Medium\n- Files: models/, train.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 09: Extract optimal policy and record convergence, Backlog 13: Develop the unified agent training pipeline, Backlog 17: Implement REINFORCE Policy Gradient updates\n\n## Tasks and acceptance criteria\n- [ ] Define a versioned serialization format that preserves full-state keys and algorithm parameters.\n- [ ] Save Value Iteration policy, Q-table and REINFORCE logits with configuration, seeds and metadata.\n- [ ] Implement consistent load functions and reject incompatible model configuration.\n- [ ] Test save/load round trips and confirm the loaded model selects the same actions on known states.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 15,
    "title": "Test and document Q-Learning",
    "owner": "Basmala",
    "assignee": "Basmala-ElKady",
    "labels": [
      "testing",
      "documentation",
      "q-learning",
      "priority: medium"
    ],
    "body": "## Objective\nVerify the TD agent and explain its learning behavior.\n\n## Ownership and scope\n- Team member: Basmala (@Basmala-ElKady)\n- Backlog ID: SNAKE-15\n- Priority: Medium\n- Files: tests/test_q_learning.py, report/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 11: Implement Q-Learning agent from scratch, Backlog 12: Implement epsilon-greedy exploration\n\n## Tasks and acceptance criteria\n- [ ] Test a hand-calculated Q-value update for ordinary and terminal transitions.\n- [ ] Test epsilon extremes, decay floor and unseen-state handling.\n- [ ] Check that truncation and true termination are treated differently.\n- [ ] Document TD equation, exploration schedule, hyperparameters and limitations; include test evidence.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 16,
    "title": "Implement tabular Softmax policy",
    "owner": "Mohamed",
    "assignee": "vexon-016",
    "labels": [
      "policy-gradient",
      "priority: high"
    ],
    "body": "## Objective\nRepresent a learnable stochastic policy over the three actions.\n\n## Ownership and scope\n- Team member: Mohamed (@vexon-016)\n- Backlog ID: SNAKE-16\n- Priority: High\n- Files: reinforce.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 03: Implement core Snake environment, Backlog 04: Implement deterministic transition interface\n\n## Tasks and acceptance criteria\n- [ ] Store three logits per full state and initialize unseen states consistently.\n- [ ] Implement numerically stable Softmax by subtracting the maximum logit.\n- [ ] Sample actions using a seeded random generator and return valid probabilities.\n- [ ] Expose greedy action selection for the shared evaluation protocol.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 17,
    "title": "Implement REINFORCE Policy Gradient updates",
    "owner": "Jowairya",
    "assignee": "jowairyakassem",
    "labels": [
      "policy-gradient",
      "priority: high"
    ],
    "body": "## Objective\nOptimize the policy directly from collected episode trajectories.\n\n## Ownership and scope\n- Team member: Jowairya (@jowairyakassem)\n- Backlog ID: SNAKE-17\n- Priority: High\n- Files: reinforce.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 16: Implement tabular Softmax policy\n\n## Tasks and acceptance criteria\n- [ ] Store state, action and reward for each trajectory; keep policy fixed while collecting the episode.\n- [ ] Compute discounted reward-to-go backwards with shared gamma.\n- [ ] Implement tabular log-Softmax gradients and the chosen discounted REINFORCE update from scratch, accumulating episode gradients before updating parameters.\n- [ ] Use configurable learning rate and document how truncated trajectories are treated (finite sampled return).\n- [ ] Log episode rewards and success without introducing a hidden reward change.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 18,
    "title": "Generate training learning curves",
    "owner": "Mohamed",
    "assignee": "vexon-016",
    "labels": [
      "visualization",
      "priority: medium"
    ],
    "body": "## Objective\nVisualize training and planning convergence clearly.\n\n## Ownership and scope\n- Team member: Mohamed (@vexon-016)\n- Backlog ID: SNAKE-18\n- Priority: Medium\n- Files: plots.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 13: Develop the unified agent training pipeline\n\n## Tasks and acceptance criteria\n- [ ] Plot reward versus episode for Q-Learning and REINFORCE; show raw or averaged traces with a documented moving-average window.\n- [ ] Plot success rate and food collected over training windows.\n- [ ] Plot Value Iteration maximum Bellman value change versus sweep on a separate axis/figure.\n- [ ] Label axes, units, method names and smoothing; export readable figures to results/.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 19,
    "title": "Generate algorithm comparison charts",
    "owner": "Mohamed",
    "assignee": "vexon-016",
    "labels": [
      "visualization",
      "evaluation",
      "priority: medium"
    ],
    "body": "## Objective\nCreate report-ready charts from held-out evaluation records.\n\n## Ownership and scope\n- Team member: Mohamed (@vexon-016)\n- Backlog ID: SNAKE-19\n- Priority: Medium\n- Files: plots.py, results/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 18: Generate training learning curves, Backlog 22: Implement unified evaluation framework\n\n## Tasks and acceptance criteria\n- [ ] Compare average total reward, success rate, food collected, steps to win and training/planning time.\n- [ ] Include variability across independent runs with clearly labeled error bars; handle missing winning episodes as N/A.\n- [ ] Use actual evaluation CSV data rather than invented values.\n- [ ] Export labeled PNG figures and keep metric definitions consistent with evaluate.py.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 20,
    "title": "Test and document REINFORCE",
    "owner": "Mohamed",
    "assignee": "vexon-016",
    "labels": [
      "testing",
      "documentation",
      "policy-gradient",
      "priority: medium"
    ],
    "body": "## Objective\nValidate Policy Gradient calculations and explain experimental limitations.\n\n## Ownership and scope\n- Team member: Mohamed (@vexon-016)\n- Backlog ID: SNAKE-20\n- Priority: Medium\n- Files: tests/test_reinforce.py, report/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 16: Implement tabular Softmax policy, Backlog 17: Implement REINFORCE Policy Gradient updates\n\n## Tasks and acceptance criteria\n- [ ] Test Softmax normalization and numerical stability.\n- [ ] Test discounted returns against a hand-calculated trajectory.\n- [ ] Test gradient direction on a simple known trajectory and reproducibility with seeds.\n- [ ] Document the policy equation, update rule, exploration through sampling, variance and truncation treatment.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 21,
    "title": "Design and implement Pygame interface",
    "owner": "Mohamed",
    "assignee": "vexon-016",
    "labels": [
      "frontend",
      "priority: high"
    ],
    "body": "## Objective\nBuild an interactive demonstration of the shared Snake environment.\n\n## Ownership and scope\n- Team member: Mohamed (@vexon-016)\n- Backlog ID: SNAKE-21\n- Priority: High\n- Files: game_ui.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 03: Implement core Snake environment\n\n## Tasks and acceptance criteria\n- [ ] Render grid, obstacles, snake head/body and food without duplicating environment logic.\n- [ ] Provide algorithm selection, start/reset controls and a trained-agent demo.\n- [ ] Display score, collected apples, cumulative reward, steps and game status.\n- [ ] Separate frame rate from environment steps and keep rendering optional.\n- [ ] Integrate saved policies when the shared loader becomes available; verify all three demos.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 22,
    "title": "Implement unified evaluation framework",
    "owner": "Jowairya",
    "assignee": "jowairyakassem",
    "labels": [
      "evaluation",
      "priority: high"
    ],
    "body": "## Objective\nCompare the trained policies fairly on held-out evaluation episodes.\n\n## Ownership and scope\n- Team member: Jowairya (@jowairyakassem)\n- Backlog ID: SNAKE-22\n- Priority: High\n- Files: evaluate.py, results/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 13: Develop the unified agent training pipeline, Backlog 14: Implement model saving and loading\n\n## Tasks and acceptance criteria\n- [ ] Use the same game rules, reward settings, initial-state schedule and evaluation scenarios for each algorithm; fix hyperparameters before final testing.\n- [ ] Run five independent training runs and 100 evaluation episodes per run as initial experimental targets, with reproducible seeds and documented actual counts.\n- [ ] Evaluate without learning or epsilon exploration; use greedy actions for all policies and document tie-breaking and deterministic repeats.\n- [ ] Calculate mean total reward, win rate, mean apples, winning steps, training/planning time and variability across runs.\n- [ ] Export raw episode records and summary CSV; document planning/model access versus sampled learning and any time-limit differences.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 23,
    "title": "Integrate all modules into the main application",
    "owner": "Basmala",
    "assignee": "Basmala-ElKady",
    "labels": [
      "integration",
      "testing",
      "priority: high"
    ],
    "body": "## Objective\nProvide a single working entry point for the complete project.\n\n## Ownership and scope\n- Team member: Basmala (@Basmala-ElKady)\n- Backlog ID: SNAKE-23\n- Priority: High\n- Files: main.py, tests/test_integration.py\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 14: Implement model saving and loading, Backlog 21: Design and implement Pygame interface, Backlog 22: Implement unified evaluation framework\n\n## Tasks and acceptance criteria\n- [ ] Expose train, evaluate and demo modes with documented algorithm selection.\n- [ ] Use shared config, environment, saved model loader and evaluation functions.\n- [ ] Handle missing/incompatible model files and invalid arguments with clear messages.\n- [ ] Run end-to-end smoke tests for all three algorithms and verify headless evaluation works without Pygame display.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 24,
    "title": "Write project README and execution instructions",
    "owner": "Jowairya",
    "assignee": "jowairyakassem",
    "labels": [
      "documentation",
      "priority: medium"
    ],
    "body": "## Objective\nMake the project reproducible for instructors and teammates.\n\n## Ownership and scope\n- Team member: Jowairya (@jowairyakassem)\n- Backlog ID: SNAKE-24\n- Priority: Medium\n- Files: README.md\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 01: Initialize project structure and dependencies, Backlog 23: Integrate all modules into the main application\n\n## Tasks and acceptance criteria\n- [ ] Explain project objective, rules, rewards, algorithms and folder structure.\n- [ ] Document setup, dependencies and exact tested commands for training, evaluation and demo.\n- [ ] Describe configuration, model artifacts, results and how to run tests.\n- [ ] Include known limitations, team responsibilities and links to the report or results as appropriate.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  },
  {
    "id": 25,
    "title": "Compile final report and submission package",
    "owner": "Jowairya",
    "assignee": "jowairyakassem",
    "labels": [
      "documentation",
      "priority: high"
    ],
    "body": "## Objective\nDeliver the required report and team contribution evidence.\n\n## Ownership and scope\n- Team member: Jowairya (@jowairyakassem)\n- Backlog ID: SNAKE-25\n- Priority: High\n- Files: report/\n- Dependencies (backlog references, not GitHub issue numbers): Backlog 05: Test and document the Snake environment, Backlog 10: Test and document Value Iteration, Backlog 15: Test and document Q-Learning, Backlog 19: Generate algorithm comparison charts, Backlog 20: Test and document REINFORCE, Backlog 22: Implement unified evaluation framework, Backlog 23: Integrate all modules into the main application, Backlog 24: Write project README and execution instructions\n\n## Tasks and acceptance criteria\n- [ ] Combine objective, game description, state/action/transition/reward/termination design and algorithm equations.\n- [ ] Document implementation, hyperparameters, evaluation protocol, actual tables/plots, comparative analysis and references.\n- [ ] Discuss state-space complexity, reward design, hyperparameters, limitations and conclusions supported by measured results.\n- [ ] Collect contribution log entries and evidence from every member; include shared testing and presentation preparation.\n- [ ] Verify source code, run instructions, game demo, written report and contribution log are ready for submission.\n\n## Definition of done\n- [ ] Relevant code or documentation is committed and linked in a pull request.\n- [ ] Applicable tests or checks pass and evidence is included.\n- [ ] Another team member reviews the work.\n- [ ] Contribution evidence is recorded for the final report.\n"
  }
]
'@ | ConvertFrom-Json
$labels = @'
{
  "environment": "1d76db",
  "dynamic-programming": "8250df",
  "q-learning": "1a7f37",
  "policy-gradient": "bf8700",
  "frontend": "bf3989",
  "evaluation": "0969da",
  "visualization": "0e8a89",
  "testing": "65758b",
  "documentation": "5468a9",
  "integration": "5319e7",
  "priority: high": "cf222e",
  "priority: medium": "bf8700"
}
'@ | ConvertFrom-Json
# Fixed ownership for all 25 tasks. No runtime assignment overrides.
if ($Preview) {
    $issues | Select-Object id, owner, assignee, title | Format-Table -AutoSize
    Write-Host 'Preview only. Nothing was changed on GitHub.'
    return
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'Install GitHub CLI first.' }
function Invoke-Gh {
    param([string[]]$Arguments)
    $output = & gh @Arguments
    if ($LASTEXITCODE -ne 0) { throw "GitHub CLI failed: gh $($Arguments -join ' ')" }
    return $output
}
$login = (Invoke-Gh -Arguments @('api','user','--jq','.login') | Out-String).Trim()
if ($login -ne 'HodaMahmoud-2005') { throw "Signed in as $login. Switch to HodaMahmoud-2005 before running this script." }
$existing = @(Invoke-Gh -Arguments @('issue','list','--repo',$repo,'--state','all','--limit','1000','--json','number,title,url,assignees,state') | Out-String | ConvertFrom-Json)
$currentLabels = @(Invoke-Gh -Arguments @('label','list','--repo',$repo,'--limit','1000','--json','name') | Out-String | ConvertFrom-Json)
foreach ($label in $labels.PSObject.Properties) {
    if ($currentLabels.name -notcontains $label.Name) {
        Invoke-Gh -Arguments @('label','create',$label.Name,'--repo',$repo,'--color',[string]$label.Value) | Out-Host
    }
}
$created = 0; $reused = 0; $warnings = 0
foreach ($issue in $issues) {
    $matches = @($existing | Where-Object { $_.title -eq $issue.title })
    if ($matches.Count -gt 1) { throw "Multiple matching issues for '$($issue.title)'. Resolve duplicates manually." }
    if ($matches.Count -eq 1) {
        $url = $matches[0].url
        $reused++
        Write-Host "EXISTS: $url - $($issue.title)"
        # Preserve existing descriptions, checkboxes and labels; sync only the ownership line and assignees.
    } else {
        $tmp = [System.IO.Path]::GetTempFileName()
        try {
            [System.IO.File]::WriteAllText($tmp, [string]$issue.body, (New-Object System.Text.UTF8Encoding($false)))
            $argsForGh = @('issue','create','--repo',$repo,'--title',[string]$issue.title,'--body-file',$tmp)
            foreach ($labelName in $issue.labels) { $argsForGh += @('--label',[string]$labelName) }
            # Create first, then assign separately so a missing/ineligible collaborator does not prevent issue creation.
            $result = Invoke-Gh -Arguments $argsForGh
            $url = ($result | Select-Object -Last 1).Trim()
            if ($url -notmatch '^https://github\.com/[^/]+/[^/]+/issues/\d+$') { throw "Unexpected issue URL: $url. Re-run to detect existing issues." }
            $created++
            Write-Host "CREATED: $url - $($issue.title)"
        } finally { Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue }
    }
    try {
        $detail = Invoke-Gh -Arguments @('issue','view',$url,'--json','number,body,assignees') | Out-String | ConvertFrom-Json
        $current = @($detail.assignees | ForEach-Object { $_.login })
        # Add and verify the intended assignee before removing previous assignees.
        if ($current -notcontains $issue.assignee) {
            Invoke-Gh -Arguments @('issue','edit',$url,'--add-assignee',[string]$issue.assignee) | Out-Host
            $detail = Invoke-Gh -Arguments @('issue','view',$url,'--json','number,body,assignees') | Out-String | ConvertFrom-Json
            $current = @($detail.assignees | ForEach-Object { $_.login })
        }
        if ($current -notcontains $issue.assignee) { throw 'New assignee not confirmed; old assignees retained.' }
        $newBody = [string]$detail.body
        $ownerLine = "- Team member: $($issue.owner) (@$($issue.assignee))"
        if ($newBody -match '(?m)^- Team member:.*$') { $newBody = $newBody -replace '(?m)^- Team member:.*$', $ownerLine }
        else { $newBody = "$ownerLine`n`n$newBody" }
        $payload = @{ assignees = @([string]$issue.assignee); body = $newBody } | ConvertTo-Json -Depth 5
        if ($current.Count -ne 1 -or $newBody -cne [string]$detail.body) {
            $tmp = [System.IO.Path]::GetTempFileName()
            try {
                [System.IO.File]::WriteAllText($tmp, $payload, (New-Object System.Text.UTF8Encoding($false)))
                Invoke-Gh -Arguments @('api','--method','PATCH',"repos/$repo/issues/$($detail.number)",'--input',$tmp,'--silent') | Out-Host
            } finally { Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue }
        }
        $verified = Invoke-Gh -Arguments @('issue','view',$url,'--json','assignees,body') | Out-String | ConvertFrom-Json
        $actual = @($verified.assignees | ForEach-Object { $_.login })
        if ($actual.Count -ne 1 -or $actual[0] -ne $issue.assignee -or $verified.body -cne $newBody) { throw 'Final ownership verification failed.' }
        Write-Host "  Assigned exclusively: $($issue.assignee)" -ForegroundColor Green
    } catch {
        $warnings++
        Write-Warning "Ownership sync pending for $url -> $($issue.assignee). $($_.Exception.Message)"
    }
}
Write-Host "Finished: $created created, $reused already existed, $warnings assignment warnings."
Write-Host "Review: https://github.com/$repo/issues"

if ($warnings -gt 0) { exit 1 }
