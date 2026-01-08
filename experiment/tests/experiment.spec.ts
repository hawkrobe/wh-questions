import { test, expect, Page } from '@playwright/test';

// Helper to click a button by text
async function clickButton(page: Page, text: string) {
  await page.getByRole('button', { name: text }).click();
}

// Helper to select radio option by label text
async function selectRadio(page: Page, labelText: string) {
  await page.getByLabel(labelText).check();
}

// Helper to wait for jsPsych trial to load
async function waitForTrial(page: Page) {
  await page.waitForSelector('.jspsych-content', { state: 'visible' });
}

// Helper to complete consent
async function completeConsent(page: Page) {
  await waitForTrial(page);
  await clickButton(page, 'I Agree');
}

// Helper to complete instructions (5 pages with visual redesign)
async function completeInstructions(page: Page) {
  for (let i = 0; i < 5; i++) {
    await waitForTrial(page);
    await clickButton(page, 'Next');
  }
}

// Helper to pass comprehension check 1
async function passComprehensionCheck1(page: Page, goalCondition: 'find' | 'avoid') {
  await waitForTrial(page);
  // Universal options - same for all conditions
  if (goalCondition === 'find') {
    await page.getByText('Prioritize finding/identifying uncontaminated').click();
  } else {
    await page.getByText('Prioritize avoiding/identifying contaminated').click();
  }
  await clickButton(page, 'Continue');

  // Click through feedback
  await waitForTrial(page);
  await clickButton(page, 'Continue');
}

// Helper to fail comprehension check 1
async function failComprehensionCheck1(page: Page) {
  await waitForTrial(page);
  await selectRadio(page, 'Test as many vials as possible');
  await clickButton(page, 'Continue');
}

// Helper to pass comprehension check 2
async function passComprehensionCheck2(page: Page) {
  await waitForTrial(page);
  await selectRadio(page, 'No, the assistant only has partial information');
  await clickButton(page, 'Continue');

  // Click through feedback
  await waitForTrial(page);
  await clickButton(page, 'Continue to Practice');
}

// Helper to complete practice trial
async function completePracticeTrial(page: Page) {
  // Practice intro
  await waitForTrial(page);
  await clickButton(page, 'Start Practice');

  // Wait for the question buttons to appear and be enabled
  await waitForTrial(page);

  // Wait for the question prompt to appear
  await page.waitForSelector('text=What would you ask your assistant?', { state: 'visible' });

  // Wait for buttons to be enabled (they're disabled for 2s)
  await page.waitForTimeout(200);

  // Click either question button - use text matching instead of class
  const contaminatedBtn = page.getByRole('button', { name: /contaminated/i });
  await contaminatedBtn.first().click();

  // Practice feedback
  await waitForTrial(page);
  await clickButton(page, 'Begin Real Trials');
}

// Helper to complete a main trial
async function completeMainTrial(page: Page, chooseFirst: boolean = true) {
  await waitForTrial(page);

  // Wait for the question prompt to appear
  await page.waitForSelector('text=What would you ask your assistant?', { state: 'visible' });

  // Wait for buttons to be enabled (they're disabled for 2s)
  await page.waitForTimeout(200);

  // Click a question button
  const contaminatedBtn = page.getByRole('button', { name: /contaminated/i });
  await contaminatedBtn.first().click();

  // Wait for ITI
  await page.waitForTimeout(600);
}

// Helper to complete strategy probe
async function completeStrategyProbe(page: Page) {
  await waitForTrial(page);
  await page.fill('textarea', 'I chose based on what seemed most helpful for my goal.');
  await clickButton(page, 'Continue');
}

// Helper to complete demographics
async function completeDemographics(page: Page) {
  await waitForTrial(page);
  await page.fill('input[name="age"]', '25');
  await page.selectOption('select[name="gender"]', 'female');
  await page.selectOption('select[name="education"]', 'bachelors');
  await page.selectOption('select[name="native_english"]', 'yes');
  await page.fill('input[name="native_language"]', 'English');
  await clickButton(page, 'Continue');
}

// Helper to complete debrief
async function completeDebrief(page: Page) {
  await waitForTrial(page);
  await clickButton(page, 'Finish and Get Completion Code');
}

// ============================================================================
// TESTS
// ============================================================================

test.describe('Experiment Flow', () => {

  test.describe('Exp 1: Goal x Base Rate', () => {

    test('completes full experiment - FIND condition', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=find&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);
      await completePracticeTrial(page);

      // Complete 12 main trials
      for (let i = 0; i < 12; i++) {
        await completeMainTrial(page);
      }

      await completeStrategyProbe(page);
      await completeDemographics(page);
      await completeDebrief(page);

      // Verify completion screen
      await expect(page.locator('text=Thank you for completing the study')).toBeVisible();
      await expect(page.locator('text=WHQPOL2024')).toBeVisible();
    });

    test('completes full experiment - AVOID condition', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=avoid&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'avoid');
      await passComprehensionCheck2(page);
      await completePracticeTrial(page);

      // Complete 12 main trials
      for (let i = 0; i < 12; i++) {
        await completeMainTrial(page);
      }

      await completeStrategyProbe(page);
      await completeDemographics(page);
      await completeDebrief(page);

      await expect(page.locator('text=WHQPOL2024')).toBeVisible();
    });

  });

  test.describe('Exp 2: Decision Structure x Goal', () => {

    test('completes singleton × find condition', async ({ page }) => {
      await page.goto('/?exp=exp2&goal=find&structure=singleton&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);
      await completePracticeTrial(page);

      // Complete 8 main trials
      for (let i = 0; i < 8; i++) {
        await completeMainTrial(page);
      }

      await completeStrategyProbe(page);
      await completeDemographics(page);
      await completeDebrief(page);

      await expect(page.locator('text=WHQPOL2024')).toBeVisible();
    });

    test('completes singleton × avoid condition', async ({ page }) => {
      await page.goto('/?exp=exp2&goal=avoid&structure=singleton&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'avoid');
      await passComprehensionCheck2(page);
      await completePracticeTrial(page);

      // Complete 8 main trials
      for (let i = 0; i < 8; i++) {
        await completeMainTrial(page);
      }

      await completeStrategyProbe(page);
      await completeDemographics(page);
      await completeDebrief(page);

      await expect(page.locator('text=WHQPOL2024')).toBeVisible();
    });

    test('completes set_id × find condition', async ({ page }) => {
      await page.goto('/?exp=exp2&goal=find&structure=set_id&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);
      await completePracticeTrial(page);

      // Complete 8 main trials
      for (let i = 0; i < 8; i++) {
        await completeMainTrial(page);
      }

      await completeStrategyProbe(page);
      await completeDemographics(page);
      await completeDebrief(page);

      await expect(page.locator('text=WHQPOL2024')).toBeVisible();
    });

    test('completes set_id × avoid condition', async ({ page }) => {
      await page.goto('/?exp=exp2&goal=avoid&structure=set_id&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'avoid');
      await passComprehensionCheck2(page);
      await completePracticeTrial(page);

      // Complete 8 main trials
      for (let i = 0; i < 8; i++) {
        await completeMainTrial(page);
      }

      await completeStrategyProbe(page);
      await completeDemographics(page);
      await completeDebrief(page);

      await expect(page.locator('text=WHQPOL2024')).toBeVisible();
    });

  });

  test.describe('Comprehension Checks', () => {

    test('allows retry on failed comprehension check 1', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=find&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);

      // Fail first attempt
      await failComprehensionCheck1(page);

      // Should see error message with 2 attempts remaining
      await waitForTrial(page);
      await expect(page.locator('text=not quite right')).toBeVisible();
      await expect(page.locator('text=2 attempt(s) remaining')).toBeVisible();

      // Click try again
      await clickButton(page, 'Try Again');

      // Now pass
      await passComprehensionCheck1(page, 'find');

      // Should proceed to check 2
      await expect(page.locator('text=Comprehension Check 2')).toBeVisible();
    });

    test('ends experiment after 3 failed attempts', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=find&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);

      // Fail 3 times
      for (let i = 0; i < 3; i++) {
        await failComprehensionCheck1(page);
        await waitForTrial(page);

        if (i < 2) {
          await clickButton(page, 'Try Again');
        }
      }

      // Should see exit message
      await expect(page.locator('text=exceeded the maximum attempts')).toBeVisible();
      await clickButton(page, 'Exit Study');

      // Experiment should end - jsPsych shows the message in the content div
      await page.waitForTimeout(500);
      const content = await page.content();
      expect(content).toContain('did not pass');
    });

  });

  test.describe('URL Parameters', () => {

    test('accepts Prolific parameters', async ({ page }) => {
      await page.goto('/?PROLIFIC_PID=TEST123&STUDY_ID=STUDY456&SESSION_ID=SESSION789&exp=exp1&goal=find');

      // Should load without error
      await expect(page.locator('text=Welcome to the Study')).toBeVisible();
    });

    test('random assignment works when no params provided', async ({ page }) => {
      await page.goto('/');

      // Should load without error and show consent
      await expect(page.locator('text=Welcome to the Study')).toBeVisible();
    });

  });

  test.describe('Trial Content', () => {

    test('shows correct goal text for FIND condition', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=find&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);

      // Check practice trial shows FIND goal
      await waitForTrial(page);
      await clickButton(page, 'Start Practice');
      await waitForTrial(page);

      await expect(page.locator('text=Find a safe vial')).toBeVisible();
    });

    test('shows correct goal text for AVOID condition', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=avoid&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'avoid');
      await passComprehensionCheck2(page);

      // Check practice trial shows AVOID goal
      await waitForTrial(page);
      await clickButton(page, 'Start Practice');
      await waitForTrial(page);

      await expect(page.locator('text=Avoid picking a contaminated vial')).toBeVisible();
    });

    test('shows correct N vials for Exp 1 (N=5)', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=find&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);

      await waitForTrial(page);
      await clickButton(page, 'Start Practice');
      await waitForTrial(page);

      // Base rate text shows "of 5" for exp1
      await expect(page.locator('text=/of 5/')).toBeVisible();
    });

    test('shows correct N vials for Exp 2 (N=4)', async ({ page }) => {
      await page.goto('/?exp=exp2&goal=find&structure=singleton&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);

      await waitForTrial(page);
      await clickButton(page, 'Start Practice');
      await waitForTrial(page);

      // Base rate text shows "of 4" for exp2
      await expect(page.locator('text=/of 4/')).toBeVisible();
    });

    test('shows set_id goal text correctly', async ({ page }) => {
      await page.goto('/?exp=exp2&goal=find&structure=set_id&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);

      await waitForTrial(page);
      await clickButton(page, 'Start Practice');
      await waitForTrial(page);

      // set_id goal shows sorting-related text
      await expect(page.locator('text=Sort all vials')).toBeVisible();
      await expect(page.locator('text=safe')).toBeVisible();
    });

  });

  test.describe('Button Randomization', () => {

    test('both question options appear on trials', async ({ page }) => {
      await page.goto('/?exp=exp1&goal=find&testMode=true');

      await completeConsent(page);
      await completeInstructions(page);
      await passComprehensionCheck1(page, 'find');
      await passComprehensionCheck2(page);

      await waitForTrial(page);
      await clickButton(page, 'Start Practice');
      await waitForTrial(page);

      // Both options should be present
      await expect(page.locator('text=Which vials are contaminated?')).toBeVisible();
      await expect(page.locator('text=Which vials are uncontaminated?')).toBeVisible();
    });

  });

});

test.describe('Data Recording', () => {

  test('logs data to console on completion', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'log') {
        consoleLogs.push(msg.text());
      }
    });

    await page.goto('/?exp=exp1&goal=find&testMode=true');

    await completeConsent(page);
    await completeInstructions(page);
    await passComprehensionCheck1(page, 'find');
    await passComprehensionCheck2(page);
    await completePracticeTrial(page);

    for (let i = 0; i < 12; i++) {
      await completeMainTrial(page);
    }

    await completeStrategyProbe(page);
    await completeDemographics(page);
    await completeDebrief(page);

    // Check that final data was logged
    const dataLog = consoleLogs.find(log => log.includes('Final experiment data'));
    expect(dataLog).toBeTruthy();
  });

});
