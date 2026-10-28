import { defineConfig, devices } from '@playwright/test';

const baseURL = 'http://127.0.0.1:8765';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: 1,
  globalSetup: './tests/e2e/setup.ts',
  reporter: [
    ['list'],
    ['html', { open: 'never' }],
    ['junit', { outputFile: 'test-results/playwright/results.xml' }],
  ],
  use: {
    baseURL,
    timezoneId: 'UTC',
    locale: 'en-US',
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});