function showImportStatus(message, type = 'loading') {
  const statusEl = el('#import-status');
  if (statusEl) {
    statusEl.textContent = message;
    statusEl.className = `import-status ${type}`;
    statusEl.style.display = 'block';
  }
}

async function exportData(type) {
  try {
    let url, filename;
    switch(type) {
      case 'labels':
        url = '/api/export-labels'; filename = 'labels.csv'; break;
      case 'full':
        url = '/api/export'; filename = 'dataset_export.zip'; break;
      case 'model':
        url = '/api/export-model'; filename = 'model_artifacts.zip'; break;
    }
    status(`📤 Exporting ${type}...`);
    const a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    status(`✅ Export started - check your downloads`);
  } catch (e) {
    status(`⚠️ Export failed: ${e.message}`);
  }
}

async function importLabels(file) {
  showImportStatus('📥 Uploading and processing file...', 'loading');
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/import-labels', { method: 'POST', body: formData });
    const result = await response.json();
    if (result.ok) {
      showImportStatus(`✅ Successfully imported ${result.imported_count} labels`, 'success');
      await loadItems();
    } else {
      showImportStatus(`❌ Import failed: ${result.msg}`, 'error');
    }
  } catch (e) {
    showImportStatus(`❌ Import error: ${e.message}`, 'error');
  }
}

function promptOrInlineConfirm(operation, sourceBtn) {
  const confirmations = {
    'reset-all-labels': 'Are you sure you want to reset ALL labels? This cannot be undone.',
    'auto-label-confident': 'Auto-label items with >90% confidence? This will apply predicted labels.',
    'backup-dataset': 'Create a backup of the current dataset?'
  };
  const message = confirmations[operation];
  if (!message) return true;
  try {
    if (confirm(message)) return true;
  } catch (_) {
    // Fall through to inline confirm
  }
  if (!sourceBtn) return false;
  // Inline confirm UX: first click arms, second click confirms within timeout
  if (!sourceBtn.dataset.confirmPending) {
    sourceBtn.dataset.confirmPending = '1';
    const originalText = sourceBtn.textContent;
    sourceBtn.dataset.originalText = originalText;
    sourceBtn.textContent = 'Click again to confirm';
    sourceBtn.classList.add('danger');
    status('Confirm required: click again to proceed.');
    // Auto-cancel after 6s
    sourceBtn._confirmTimer && clearTimeout(sourceBtn._confirmTimer);
    sourceBtn._confirmTimer = setTimeout(() => {
      if (sourceBtn.dataset.confirmPending) {
        sourceBtn.textContent = sourceBtn.dataset.originalText || originalText;
        delete sourceBtn.dataset.confirmPending;
        delete sourceBtn.dataset.originalText;
        status('');
      }
    }, 6000);
    return false;
  }
  // Second click within window: proceed
  sourceBtn.textContent = sourceBtn.dataset.originalText || sourceBtn.textContent;
  delete sourceBtn.dataset.confirmPending;
  delete sourceBtn.dataset.originalText;
  sourceBtn._confirmTimer && clearTimeout(sourceBtn._confirmTimer);
  return true;
}

async function performBulkOperation(operation, sourceBtn) {
  console.log('performBulkOperation called with:', operation, sourceBtn);
  status(`🔄 Starting ${operation}...`);
  const okToProceed = promptOrInlineConfirm(operation, sourceBtn);
  console.log('Confirmation result:', okToProceed);
  if (!okToProceed) return;
  try {
    status(`⚡ Performing ${operation.replace('-', ' ')}...`);
    const result = await api(`/bulk-${operation}`, { method: 'POST' });
    if (result.ok) {
      status(`✅ ${operation.replace('-', ' ')} completed: ${result.msg}`);
      try { alert(`✅ ${result.msg || 'Operation completed'}`); } catch(_) {}
      const modal = document.querySelector('#bulkOpsModal');
      if (modal && !modal.classList.contains('hidden')) closeModal('#bulkOpsModal');
      await loadItems();
    } else {
      status(`⚠️ Operation failed: ${result.msg}`);
      try { alert(`❌ Operation failed: ${result.msg || 'Unknown error'}`); } catch(_) {}
    }
  } catch (e) {
    status(`⚠️ Operation error: ${e.message}`);
    try { alert(`❌ Operation error: ${e.message}`); } catch(_) {}
  }
}


