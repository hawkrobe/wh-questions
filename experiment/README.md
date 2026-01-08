# Wh-Question Polarity Choice Experiment

jsPsych 8 implementation of two experiments investigating how questioners choose between "Which are contaminated?" vs "Which are uncontaminated?" questions.

## Quick Start

1. Open `index.html` in a browser, or serve locally:
   ```bash
   python -m http.server 8000
   # Then visit http://localhost:8000/experiment/
   ```

2. The experiment will randomly assign conditions unless specified via URL parameters.

## URL Parameters

### Prolific Integration
- `PROLIFIC_PID` - Participant ID from Prolific
- `STUDY_ID` - Study ID from Prolific
- `SESSION_ID` - Session ID from Prolific

### Condition Assignment
- `exp` - Experiment version: `exp1` or `exp2`
- `goal` - Goal condition: `find` or `avoid`
- `structure` - Decision structure (Exp 2 only): `singleton` or `set_id`

### Example URLs

**Exp 1 - Find condition:**
```
index.html?exp=exp1&goal=find
```

**Exp 1 - Avoid condition:**
```
index.html?exp=exp1&goal=avoid
```

**Exp 2 - Singleton × Find:**
```
index.html?exp=exp2&goal=find&structure=singleton
```

**Exp 2 - Set ID × Avoid:**
```
index.html?exp=exp2&goal=avoid&structure=set_id
```

**Full Prolific URL example:**
```
https://yourserver.com/experiment/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&exp=exp1&goal=find
```

## Experiment Structure

### Exp 1: Goal × Base Rate
- **Between-subjects:** Goal (FIND vs AVOID)
- **Within-subjects:** Base rate (20%, 50%, 80%) × 4 trials each = 12 trials
- Decision structure: Singleton only

### Exp 2: Decision Structure × Goal
- **Between-subjects:** Decision Structure (singleton vs set_id) × Goal (FIND vs AVOID)
- 8 trials at 50% base rate

## Trial Flow

1. **Welcome/Consent** - IRB-compliant consent form
2. **Instructions** (4 pages) - Task overview, assistant info, question options
3. **Comprehension Checks**
   - Goal check (must pass, max 3 attempts)
   - Assistant knowledge check (must pass)
4. **Practice Trial** - With feedback
5. **Main Trials** - 12 (Exp 1) or 8 (Exp 2) trials
   - 2s minimum viewing time before response
   - 500ms ITI
6. **Strategy Probe** - Free response about reasoning
7. **Demographics** - Age, gender, education, native language
8. **Debrief** - Study explanation

## Data Format

Data is saved as JSON with the following structure:

```json
{
  "prolific_pid": "...",
  "study_id": "...",
  "session_id": "...",
  "exp_version": "exp1",
  "goal_condition": "find",
  "decision_structure": "singleton",
  "completion_time": "2024-...",
  "trials": [
    {
      "trial_type": "main_trial",
      "trial_id": "exp1_br0.5_rep1",
      "base_rate": 0.5,
      "trial_index": 0,
      "button_order": ["Which vials are contaminated?", "Which vials are uncontaminated?"],
      "chosen_question": "Which vials are uncontaminated?",
      "chose_which_contaminated": false,
      "chose_which_uncontaminated": true,
      "response": 1,
      "rt": 3456
    }
  ]
}
```

## Key Data Fields

For each main trial:
- `base_rate` - Contamination rate (0.2, 0.5, or 0.8)
- `chosen_question` - Full text of chosen question
- `chose_which_contaminated` - Boolean: chose "Which are contaminated?"
- `chose_which_uncontaminated` - Boolean: chose "Which are uncontaminated?"
- `rt` - Response time in milliseconds
- `button_order` - Randomized order of buttons (for checking position effects)

## Analysis Notes

### Primary DV
`chose_which_uncontaminated` (boolean) - aligns with model predictions where:
- FIND goal → higher P(ask "Which uncontaminated?")
- AVOID goal → lower P(ask "Which uncontaminated?")

### Expected Patterns

**Exp 1 (singleton):**
- Strong Goal × Base Rate interaction
- Goal effect largest at 50% base rate

**Exp 2:**
- Goal effect large for singleton
- Goal effect null/attenuated for set_id
- Decision Structure × Goal interaction

## Counterbalancing

For proper counterbalancing with Prolific, set up 4 studies for Exp 2:
1. `?exp=exp2&goal=find&structure=singleton`
2. `?exp=exp2&goal=avoid&structure=singleton`
3. `?exp=exp2&goal=find&structure=set_id`
4. `?exp=exp2&goal=avoid&structure=set_id`

Or 2 studies for Exp 1:
1. `?exp=exp1&goal=find`
2. `?exp=exp1&goal=avoid`

## Server Integration

To save data to your server, uncomment the `fetch` call in `on_finish` and configure your endpoint:

```javascript
fetch('/save_data', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(finalData)
});
```

## Dependencies

All loaded from CDN (no local installation required):
- jsPsych 8.0.2
- @jspsych/plugin-html-button-response 2.0.0
- @jspsych/plugin-html-keyboard-response 2.0.0
- @jspsych/plugin-survey-html-form 2.0.0
- @jspsych/plugin-survey-text 2.0.0
- @jspsych/plugin-survey-multi-choice 2.0.0
- @jspsych/plugin-instructions 2.0.0
- @jspsych/plugin-call-function 2.0.0
- @jspsych/plugin-preload 2.0.0
