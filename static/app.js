(function () {
  // Drag & drop file upload handler
  const input = document.getElementById('dataset-input');
  const dz = document.getElementById('dropzone');
  const text = document.getElementById('dropzone-text');

  if (input && dz && text) {
    input.addEventListener('change', () => {
      if (input.files[0]) text.textContent = input.files[0].name;
    });

    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
      dz.classList.add('drag');
    });

    dz.addEventListener('dragleave', () => dz.classList.remove('drag'));

    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('drag');
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        text.textContent = e.dataTransfer.files[0].name;
      }
    });
  }

  // Upload submit loading state
  const form = document.getElementById('upload-form');
  if (form) {
    form.addEventListener('submit', () => {
      const b = document.getElementById('submit-btn');
      if (b) {
        b.disabled = true;
        b.textContent = 'Inspecting dataset…';
      }
    });
  }

  // Live training progress polling
  if (typeof PROGRESS_API !== 'undefined') {
    const fill = document.getElementById('progress-fill');
    const pct = document.getElementById('progress-pct');
    const desc = document.getElementById('progress-desc');
    const meta = document.getElementById('trial-meta');
    const log = document.getElementById('live-log');
    const badge = document.getElementById('training-status-badge');

    const liveEpoch = document.getElementById('live-epoch');
    const liveEpochSub = document.getElementById('live-epoch-sub');
    const liveTrainLoss = document.getElementById('live-train-loss');
    const liveTrainAcc = document.getElementById('live-train-acc');
    const liveTime = document.getElementById('live-time');
    const liveRemaining = document.getElementById('live-remaining');

    let seenLogs = 0;

    async function poll() {
      try {
        const r = await fetch(PROGRESS_API);
        if (!r.ok) {
          setTimeout(poll, 2000);
          return;
        }
        const d = await r.json();

        if (fill) fill.style.width = (d.pct || 0) + '%';
        if (pct) pct.textContent = (d.pct || 0) + '%';
        if (desc) desc.textContent = d.description || 'Training in progress…';
        if (meta) meta.textContent = (d.model_index || 0) + ' / ' + (d.total_models || 1) + ' models completed';
        if (badge) badge.textContent = (d.status || 'RUNNING').toUpperCase();

        if (liveEpoch && d.current_epoch) {
          liveEpoch.textContent = 'Epoch ' + d.current_epoch + ' / ' + d.total_epochs;
        }
        if (liveEpochSub && d.current_model) {
          liveEpochSub.textContent = d.current_model;
        }
        if (liveTrainLoss && d.train_loss !== undefined && d.train_loss !== null) {
          liveTrainLoss.textContent = typeof d.train_loss === 'number' ? d.train_loss.toFixed(4) : d.train_loss;
        }
        if (liveTrainAcc && d.train_acc !== undefined && d.train_acc !== null) {
          liveTrainAcc.textContent = (d.train_acc * 100).toFixed(1) + '%';
        }
        if (liveTime && d.elapsed_seconds !== undefined) {
          liveTime.textContent = d.elapsed_seconds + 's';
        }
        if (liveRemaining && d.remaining_seconds !== undefined) {
          liveRemaining.textContent = d.remaining_seconds > 0 ? '~' + d.remaining_seconds + 's remaining' : 'Finishing…';
        }

        if (log && d.logs && Array.isArray(d.logs)) {
          if (d.logs.length > seenLogs) {
            for (let i = seenLogs; i < d.logs.length; i++) {
              const div = document.createElement('div');
              div.innerHTML = '<i></i> ' + d.logs[i];
              log.appendChild(div);
            }
            seenLogs = d.logs.length;
            log.scrollTop = log.scrollHeight;
          }
        }

        if (d.status === 'done') {
          if (pct) pct.textContent = '100%';
          if (fill) fill.style.width = '100%';
          if (desc) desc.textContent = 'Training finished! Loading results…';
          setTimeout(() => { location.href = RESULTS_URL; }, 800);
          return;
        }

        if (d.status === 'error') {
          if (desc) desc.textContent = 'Training failed: ' + (d.error || 'Unknown error');
          if (badge) {
            badge.textContent = 'FAILED';
            badge.style.color = 'var(--danger)';
          }
          return;
        }

        if (d.status === 'cancelled') {
          if (desc) desc.textContent = 'Training was cancelled.';
          if (badge) {
            badge.textContent = 'CANCELLED';
            badge.style.color = 'var(--danger)';
          }
          return;
        }

        setTimeout(poll, 1000);
      } catch (err) {
        setTimeout(poll, 2000);
      }
    }

    poll();

    const cancel = document.getElementById('cancel-btn');
    if (cancel) {
      cancel.onclick = async () => {
        cancel.disabled = true;
        cancel.textContent = 'Cancelling…';
        await fetch(CANCEL_API, { method: 'POST' });
      };
    }
  }
})();
