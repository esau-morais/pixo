import { test, expect, FIXTURES } from './fixtures';

test.describe('Keyboard Navigation', () => {
	test('should open file picker with Cmd/Ctrl+O', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileChooserPromise = page.waitForEvent('filechooser', { timeout: 5000 });
		await page.keyboard.press('Control+o');
		const fileChooser = await fileChooserPromise;
		expect(fileChooser).toBeTruthy();
		await fileChooser.setFiles([]);
	});

	test('should close single view with Escape when only one file', async ({ page, waitForWasm, uploadAndWaitForCompression }) => {
		await page.goto('/');
		await waitForWasm();
		await uploadAndWaitForCompression(FIXTURES.PNG);
		await expect(page.getByTestId('single-view')).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'drop');
	});

	test('should go back to list view with Escape when multiple files', async ({ page, waitForWasm }) => {
		await page.goto('/');
		await waitForWasm();
		const fileInput = page.getByTestId('file-input');
		await fileInput.setInputFiles([FIXTURES.PNG, FIXTURES.JPEG]);
		await expect(page.getByTestId('download-button')).toBeVisible({ timeout: 120000 });
		await page.getByTestId('job-row').first().click();
		await expect(page.getByTestId('single-view')).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(page.getByTestId('main-content')).toHaveAttribute('data-view-mode', 'list');
		await expect(page.getByTestId('list-view')).toBeVisible();
	});
});
