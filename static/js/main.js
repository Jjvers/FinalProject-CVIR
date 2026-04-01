/**
 * Smart Door Lock System - Main JavaScript
 * Handles fire alarm, status polling, and global UI interactions.
 */

// ─── Fire Alarm ────────────────────────────────────────────────

function triggerFireAlarm() {
    if (!confirm('⚠️ ACTIVATE FIRE ALARM?\n\nThis will unlock ALL doors for emergency evacuation!')) {
        return;
    }

    fetch('/api/fire_alarm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'activate' })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showFireBanner(true);
        }
    })
    .catch(err => console.error('Fire alarm error:', err));
}

function deactivateFireAlarm() {
    fetch('/api/fire_alarm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'deactivate' })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showFireBanner(false);
        }
    })
    .catch(err => console.error('Deactivate error:', err));
}

function showFireBanner(active) {
    const banner = document.getElementById('fireAlertBanner');
    if (banner) {
        if (active) {
            banner.classList.add('active');
            document.body.style.paddingTop = '60px';
        } else {
            banner.classList.remove('active');
            document.body.style.paddingTop = '0';
        }
    }
}

// ─── Check fire alarm on page load ─────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            if (data.fire_alarm_active) {
                showFireBanner(true);
            }
            // Update door status
            const statusText = document.getElementById('doorStatusText');
            const statusDot = document.getElementById('statusDot');
            if (statusText && data.door_state) {
                if (data.door_state === 'UNLOCKED') {
                    statusText.textContent = 'Door: Unlocked';
                    if (statusDot) statusDot.style.background = 'var(--accent-orange)';
                } else {
                    statusText.textContent = 'Door: Locked';
                }
            }
        })
        .catch(() => {});
});
