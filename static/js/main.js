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
    if (active) {
        if(banner) banner.classList.add('active');
        document.body.classList.add('fire-mode');
    } else {
        if(banner) banner.classList.remove('active');
        document.body.classList.remove('fire-mode');
    }
}

// ─── Global System Status Poller ───────────────────────────────

function pollGlobalStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(data => {
            showFireBanner(data.fire_alarm_active === true);
            
            // Update door status badge if present
            const statusText = document.getElementById('doorStatusText');
            const statusDot = document.getElementById('statusDot');
            if (statusText && data.door_state) {
                if (data.door_state === 'UNLOCKED') {
                    statusText.textContent = 'Door: Unlocked';
                    if (statusDot) statusDot.style.background = 'var(--accent-orange)';
                } else {
                    statusText.textContent = 'Door: Locked';
                    if (statusDot) statusDot.style.background = 'var(--accent-green)';
                }
            }
        })
        .catch(() => {});
}

// Polling status background check every 2.5 seconds (Fast reaction to fire)
setInterval(pollGlobalStatus, 2500);

document.addEventListener('DOMContentLoaded', () => {
    pollGlobalStatus();
});
