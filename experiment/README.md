# Wh-Question Polarity Choice Experiment

jsPsych implementation of two experiments investigating how questioners choose between "Which are contaminated?" vs "Which are uncontaminated?" questions.

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

**Full Prolific URL example:**
```
https://yourserver.com/experiment/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&exp=exp1&goal=find
```
