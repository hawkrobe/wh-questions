// =============================================================================
// SHARED EXPERIMENT COMPONENTS
// Common functions and templates used by exp1.html and exp2.html
// =============================================================================

// -----------------------------------------------------------------------------
// VIAL RENDERING
// -----------------------------------------------------------------------------

function renderVial(id, state = 'unknown', showLabel = true) {
    const testedClass = (state === 'safe' || state === 'danger') ? 'tested' : '';
    return `
        <div class="vial-container">
            <div class="vial ${state} ${testedClass}">
                <div class="vial-body">
                    <div class="vial-liquid"></div>
                </div>
                <div class="vial-status"></div>
            </div>
            ${showLabel ? `<div class="vial-label">${id}</div>` : ''}
        </div>
    `;
}

function renderVialRack(nVials, states = null) {
    if (!states) {
        states = Array(nVials).fill('unknown');
    }
    const vials = states.map((state, i) => renderVial(i + 1, state)).join('');
    return `<div class="vial-rack">${vials}</div>`;
}

// -----------------------------------------------------------------------------
// PROPORTION BAR
// -----------------------------------------------------------------------------

function renderProportionBar(baseRate, nVials) {
    const nContaminated = Math.round(baseRate * nVials);
    const nSafe = nVials - nContaminated;
    const contaminatedPct = baseRate * 100;
    const safePct = 100 - contaminatedPct;

    return `
        <div class="proportion-container">
            <div class="proportion-label">Expected contamination rate:</div>
            <div class="proportion-bar">
                <div class="proportion-segment contaminated" style="width: ${contaminatedPct}%;">
                    ${nContaminated > 0 ? `${nContaminated} ☠` : ''}
                </div>
                <div class="proportion-segment safe" style="width: ${safePct}%;">
                    ${nSafe > 0 ? `${nSafe} ✓` : ''}
                </div>
            </div>
            <div class="proportion-legend">
                <div class="legend-item">
                    <div class="legend-dot contaminated"></div>
                    <span>Contaminated (${nContaminated})</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot safe"></div>
                    <span>Safe (${nSafe})</span>
                </div>
            </div>
        </div>
    `;
}

// -----------------------------------------------------------------------------
// UI HELPERS
// -----------------------------------------------------------------------------

function disableButtonsTemporarily(duration) {
    const buttons = document.querySelectorAll('.jspsych-btn');
    buttons.forEach(btn => {
        btn.disabled = true;
        btn.style.opacity = '0.5';
    });
    setTimeout(() => {
        buttons.forEach(btn => {
            btn.disabled = false;
            btn.style.opacity = '1';
        });
    }, duration);
}

// -----------------------------------------------------------------------------
// CONSENT FORM
// -----------------------------------------------------------------------------

const CONSENT_HTML = `
    <h2 style="text-align: center;">STANFORD UNIVERSITY</h2>
    <h3 style="text-align: center;">Research Information Sheet</h3>
    <div class="instructions-box" style="font-size: 15px; line-height: 1.5;">
        <p><strong>Protocol Director:</strong> Robert Hawkins<br>
        <strong>Protocol Title:</strong> Communication and social cognition in natural audiovisual contexts<br>
        <strong>IRB#</strong> 77226</p>

        <p><strong>DESCRIPTION:</strong> You are invited to participate in a research study about language and communication. The purpose of the research is to understand how people formulate questions to gather information in uncertain situations. This research will be conducted through the Prolific platform, including participants from the US, UK, and Canada.</p>

        <p><strong>TIME INVOLVEMENT:</strong> The task will last the amount of time advertised on Prolific. You are free to withdraw from the study at any time.</p>

        <p><strong>RISKS AND BENEFITS:</strong> Study data will be stored securely, in compliance with Stanford University standards, minimizing the risk of confidentiality breach. This study advances our scientific understanding of how people communicate and make decisions. We cannot and do not guarantee or promise that you will receive any benefits from this study.</p>

        <p><strong>PAYMENTS:</strong> You will receive payment in the amount advertised on Prolific. If you do not complete this study, you will receive prorated payment based on the time that you have spent.</p>

        <p><strong>PARTICIPANT'S RIGHTS:</strong> If you have read this form and have decided to participate in this project, please understand your participation is voluntary and you have the right to withdraw your consent or discontinue participation at any time without penalty or loss of benefits to which you are otherwise entitled. The alternative is not to participate. You have the right to refuse to answer particular questions. The results of this research study may be presented at scientific or professional meetings or published in scientific journals. Your individual privacy will be maintained in all published and written data resulting from the study. In accordance with scientific norms, the data from this study may be used or shared with other researchers for future research (after removing personally identifying information) without additional consent from you.</p>

        <p><strong>CONTACT INFORMATION:</strong><br>
        <em>Questions:</em> If you have any questions, concerns or complaints about this research, its procedures, risks and benefits, contact the Protocol Director, Robert Hawkins (rdhawkins@stanford.edu).<br><br>
        <em>Independent Contact:</em> If you are not satisfied with how this study is being conducted, or if you have any concerns, complaints, or general questions about the research or your rights as a participant, please contact the Stanford Institutional Review Board (IRB) to speak to someone independent of the research team at 650-723-2480 or toll free at 1-866-680-2906, or email at irb2-manager@stanford.edu. You can also write to the Stanford IRB, Stanford University, 1705 El Camino Real, Palo Alto, CA 94306.</p>

        <p style="text-align: center; margin-top: 20px;"><strong>Please save or print a copy of this page for your records.<br>If you agree to participate in this research, please click "Continue".</strong></p>
    </div>
`;

// -----------------------------------------------------------------------------
// DEMOGRAPHICS FORM
// -----------------------------------------------------------------------------

const DEMOGRAPHICS_HTML = `
    <h2>Demographics</h2>
    <p>Please answer the following questions about yourself.</p>

    <div style="margin: 20px 0;">
        <label for="age"><strong>Age:</strong></label><br>
        <input type="number" id="age" name="age" min="18" max="100" required
               style="padding: 8px; width: 100px; margin-top: 5px;">
    </div>

    <div style="margin: 20px 0;">
        <label><strong>Gender:</strong></label><br>
        <select name="gender" required style="padding: 8px; width: 200px; margin-top: 5px;">
            <option value="">-- Select --</option>
            <option value="female">Female</option>
            <option value="male">Male</option>
            <option value="non-binary">Non-binary</option>
            <option value="other">Other</option>
            <option value="prefer_not">Prefer not to say</option>
        </select>
    </div>

    <div style="margin: 20px 0;">
        <label><strong>Highest level of education completed:</strong></label><br>
        <select name="education" required style="padding: 8px; width: 300px; margin-top: 5px;">
            <option value="">-- Select --</option>
            <option value="high_school">High school or equivalent</option>
            <option value="some_college">Some college</option>
            <option value="associates">Associate's degree</option>
            <option value="bachelors">Bachelor's degree</option>
            <option value="masters">Master's degree</option>
            <option value="doctorate">Doctorate or professional degree</option>
            <option value="other">Other</option>
        </select>
    </div>

    <div style="margin: 20px 0;">
        <label><strong>Is English your native language?</strong></label><br>
        <select name="native_english" required style="padding: 8px; width: 200px; margin-top: 5px;">
            <option value="">-- Select --</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
        </select>
    </div>

    <div style="margin: 20px 0;">
        <label for="native_language"><strong>What is your native language?</strong></label><br>
        <input type="text" id="native_language" name="native_language" required
               style="padding: 8px; width: 250px; margin-top: 5px;">
    </div>
`;

// -----------------------------------------------------------------------------
// DEBRIEF
// -----------------------------------------------------------------------------

const DEBRIEF_HTML = `
    <h2>Study Complete - Debriefing</h2>
    <div class="instructions-box">
        <p><strong>Thank you for participating!</strong></p>

        <p><strong>Purpose of this study:</strong></p>
        <p>This study investigates how people choose to phrase questions when seeking information.
        Specifically, we are interested in whether people's goals influence which question they ask—for example,
        whether asking "Which are contaminated?" vs. "Which are uncontaminated?" depends on what the person
        wants to accomplish.</p>

        <p><strong>What we're studying:</strong></p>
        <p>We hypothesize that people strategically choose question framing to maximize the usefulness of
        the answer for their specific decision problem. Your responses will help us understand the
        cognitive processes underlying question formulation.</p>

        <p><strong>Confidentiality:</strong></p>
        <p>Your responses are completely anonymous and will only be used for research purposes.
        No identifying information has been collected.</p>

        <p>If you have any questions about this research, please contact sylee423@stanford.edu or rdhawkins@stanford.edu.</p>
    </div>
`;

// -----------------------------------------------------------------------------
// COMPREHENSION CHECK BUILDERS
// -----------------------------------------------------------------------------

function createComprehensionCheck1(jsPsych, goalCondition, decisionStructure) {
    const maxAttempts = 3;
    let attempts = 0;
    let passed = false;

    // Options depend on decision structure
    const optionsBase = decisionStructure === 'singleton'
        ? [
            'Find an uncontaminated vial to use',
            'Find a contaminated vial to examine',
            'Test as many vials as possible',
            'Ask the assistant multiple questions'
        ]
        : [
            'Help Dr. Smith\'s experiment succeed',
            'Help the lab pass the safety inspection',
            'Sort vials as quickly as possible',
            'Ask the assistant multiple questions'
        ];

    const correctAnswer = goalCondition === 'uncont' ? optionsBase[0] : optionsBase[1];
    let shuffledOptions = [];

    const check = {
        type: jsPsychSurveyMultiChoice,
        questions: function() {
            shuffledOptions = jsPsych.randomization.shuffle([...optionsBase]);
            return [{
                prompt: '<strong>Comprehension Check 1:</strong> What is your <strong>main goal</strong> in this task?',
                name: 'goal_check',
                options: shuffledOptions,
                required: true
            }];
        },
        data: { trial_type: 'comprehension_check_1' },
        on_finish: function(data) {
            attempts++;
            const response = data.response.goal_check;
            data.correct = response === correctAnswer;
            data.option_order = shuffledOptions;
            passed = data.correct;
        }
    };

    const feedback = {
        type: jsPsychHtmlButtonResponse,
        stimulus: function() {
            if (passed) {
                return `<p class="check-correct">✓ Correct! Let's continue to the next check.</p>`;
            } else {
                const attemptsLeft = maxAttempts - attempts;
                if (attemptsLeft > 0) {
                    let reminderText;
                    if (decisionStructure === 'singleton') {
                        reminderText = goalCondition === 'uncont'
                            ? 'find an uncontaminated vial to use'
                            : 'find a contaminated vial to examine';
                    } else {
                        reminderText = goalCondition === 'uncont'
                            ? 'help Dr. Smith\'s experiment succeed'
                            : 'help the lab pass the safety inspection';
                    }
                    return `<p class="check-incorrect">That's not quite right.</p>
                            <p>Remember: Your goal is to <strong>${reminderText}</strong>.</p>
                            <p>You have ${attemptsLeft} attempt(s) remaining.</p>`;
                } else {
                    return `<p class="check-incorrect">You have exceeded the maximum attempts.</p>
                            <p>Unfortunately, you cannot continue with this study.</p>`;
                }
            }
        },
        choices: function() {
            if (passed) return ['Continue'];
            if (attempts >= maxAttempts) return ['Exit Study'];
            return ['Try Again'];
        },
        data: { trial_type: 'comprehension_feedback_1' },
        on_finish: function(data) {
            if (!passed && attempts >= maxAttempts) {
                jsPsych.endExperiment('Thank you for your time. You did not pass the comprehension checks.');
            }
        }
    };

    return {
        timeline: [check, feedback],
        loop_function: () => !passed && attempts < maxAttempts
    };
}

function createComprehensionCheck2(jsPsych) {
    const maxAttempts = 3;
    let attempts = 0;
    let passed = false;

    const optionsBase = [
        'Yes, the assistant knows everything.',
        'No, the assistant only has partial information.',
        'The assistant knows nothing.',
        'It depends on the trial.'
    ];
    const correctAnswer = 'No, the assistant only has partial information.';
    let shuffledOptions = [];

    const check = {
        type: jsPsychSurveyMultiChoice,
        questions: function() {
            shuffledOptions = jsPsych.randomization.shuffle([...optionsBase]);
            return [{
                prompt: '<strong>Comprehension Check 2:</strong> Does your lab assistant know the status of ALL the vials?',
                name: 'assistant_check',
                options: shuffledOptions,
                required: true
            }];
        },
        data: { trial_type: 'comprehension_check_2' },
        on_finish: function(data) {
            attempts++;
            const response = data.response.assistant_check;
            data.correct = response === correctAnswer;
            data.option_order = shuffledOptions;
            passed = data.correct;
        }
    };

    const feedback = {
        type: jsPsychHtmlButtonResponse,
        stimulus: function() {
            if (passed) {
                return `<p class="check-correct">✓ Correct! You're ready to begin.</p>`;
            } else {
                const attemptsLeft = maxAttempts - attempts;
                if (attemptsLeft > 0) {
                    return `<p class="check-incorrect">That's not quite right.</p>
                            <p>Remember: The assistant knows about <strong>some but not all</strong> of the vials, so they only have <strong>partial information</strong>.</p>
                            <p>You have ${attemptsLeft} attempt(s) remaining.</p>`;
                } else {
                    return `<p class="check-incorrect">You have exceeded the maximum attempts.</p>
                            <p>Unfortunately, you cannot continue with this study.</p>`;
                }
            }
        },
        choices: function() {
            if (passed) return ['Begin Trial'];
            if (attempts >= maxAttempts) return ['Exit Study'];
            return ['Try Again'];
        },
        data: { trial_type: 'comprehension_feedback_2' },
        on_finish: function(data) {
            if (!passed && attempts >= maxAttempts) {
                jsPsych.endExperiment('Thank you for your time. You did not pass the comprehension checks.');
            }
        }
    };

    return {
        timeline: [check, feedback],
        loop_function: () => !passed && attempts < maxAttempts
    };
}

// -----------------------------------------------------------------------------
// COMPLETION SCREEN
// -----------------------------------------------------------------------------

function showCompletionScreen(jsPsych, finalData, completionCode, prolificPID) {
    const dataBlob = new Blob([JSON.stringify(finalData, null, 2)], {type: 'application/json'});
    const dataUrl = URL.createObjectURL(dataBlob);

    jsPsych.getDisplayElement().innerHTML = `
        <div style="text-align: center; padding: 50px;">
            <h2>Thank you for completing the study!</h2>
            <p>Your completion code is:</p>
            <h1 style="background: #f0f0f0; padding: 20px; border-radius: 10px;">${completionCode}</h1>
            <p>Please copy this code and paste it into Prolific to receive your payment.</p>
            <p style="margin-top: 30px; color: #666;">You may now close this window.</p>
            <p style="margin-top: 20px;">
                <a href="${dataUrl}" download="wh_questions_data_${prolificPID}_${Date.now()}.json"
                   style="color: #666; font-size: 12px;">[Download data (for testing)]</a>
            </p>
        </div>
    `;

    console.log('Final experiment data:', JSON.stringify(finalData, null, 2));
}
