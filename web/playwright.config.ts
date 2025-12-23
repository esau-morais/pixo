import { defineConfig, devices } from 'playwright/test';

export default defineConfig({
	testDir: './e2e',
	fullyParallel: true,
	forbidOnly: !!process.env.CI,
	retries: process.env.CI ? 2 : 1,
	workers: process.env.CI ? 1 : undefined,
	reporter: process.env.CI 
		? [['html', { open: 'never' }], ['github']]
		: [['html', { open: 'on-failure' }]],
	use: {
		baseURL: process.env.BASE_URL || 'http://localhost:4173',
		trace: 'on-first-retry',
		screenshot: 'only-on-failure',
		video: 'on-first-retry',
		actionTimeout: 10000,
		navigationTimeout: 30000,
	},
	timeout: 60000,
	expect: {
		timeout: 10000,
	},
	projects: [
		{
			name: 'chromium',
			use: { 
				...devices['Desktop Chrome'],
				viewport: { width: 1280, height: 720 },
			},
		},
	],
	webServer: {
		command: 'npm run preview',
		url: 'http://localhost:4173',
		reuseExistingServer: !process.env.CI,
		timeout: 120000,
	},
});
