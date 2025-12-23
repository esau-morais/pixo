import { test, expect, FIXTURES } from './fixtures';

test.describe('Multi-File Handling', () => {
	test('should switch to list view when uploading multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'list');
		await expect(page.getByTestId('list-view')).toBeVisible();
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		await expect(page.getByTestId('file-count')).toHaveText('2 files');
	});

	test('should display all jobs in list view', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		const jobRows = page.getByTestId('job-row');
		await expect(jobRows).toHaveCount(2);
		const jobNames = page.getByTestId('job-name');
		const names = await jobNames.allTextContents();
		expect(names).toContain('playground.png');
		expect(names).toContain('multi-agent.jpg');
	});

	test('should show compression status for all jobs', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		const jobRows = page.getByTestId('job-row');
		for (const row of await jobRows.all()) {
			await expect(row).toHaveAttribute('data-job-status', 'done', { timeout: 60000 });
		}
		const savings = page.getByTestId('job-savings');
		await expect(savings).toHaveCount(2);
	});

	test('should navigate from list view to single view', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		const firstJob = page.getByTestId('job-row').first();
		await firstJob.click();
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'single');
		await expect(page.getByTestId('single-view')).toBeVisible();
		await expect(page.getByTestId('back-button')).toBeVisible();
	});

	test('should navigate back from single view to list view', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		await page.getByTestId('job-row').first().click();
		await expect(page.getByTestId('single-view')).toBeVisible();
		await page.getByTestId('back-button').click();
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'list');
		await expect(page.getByTestId('list-view')).toBeVisible();
	});

	test('should offer zip download for multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		await expect(page.getByTestId('download-button')).toHaveText('Download All (.zip)');
	});

	test('should trigger zip download for multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		const downloadPromise = page.waitForEvent('download', { timeout: 30000 });
		await page.getByTestId('download-button').click();
		const download = await downloadPromise;
		expect(download.suggestedFilename()).toBe('compressed-images.zip');
	});

	test('should clear all files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('list-view')).toBeVisible();
		await page.getByTestId('clear-all-button').click();
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
		await expect(page.getByTestId('drop-zone')).toBeVisible();
	});

	test('should show aggregate stats for multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		await expect(page.getByTestId('total-original-size')).toBeVisible();
		await expect(page.getByTestId('total-compressed-size')).toBeVisible();
		await expect(page.getByTestId('total-savings')).toBeVisible();
	});
});
