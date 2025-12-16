// Theme toggle
function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === 'light') {
    root.setAttribute('data-theme', 'light');
    document.body.classList.add('light-theme');
    document.body.classList.remove('dark-theme');
    document.documentElement.style.colorScheme = 'light';
  } else {
    root.removeAttribute('data-theme');
    document.body.classList.add('dark-theme');
    document.body.classList.remove('light-theme');
    document.documentElement.style.colorScheme = 'dark';
    theme = 'dark';
  }
  const icon = document.getElementById('themeIcon');
  if (icon) icon.textContent = theme === 'dark' ? '🌙' : '☀️';
  localStorage.setItem('wa-theme', theme);
}

function initTheme() {
  // Check for saved user preference, if any, on load
  const savedTheme = localStorage.getItem('wa-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  // If no saved theme, use system preference, otherwise use saved theme
  const theme = savedTheme || (prefersDark ? 'dark' : 'light');
  
  // Apply the theme
  applyTheme(theme);
  
  // Set up theme toggle button
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const currentTheme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      applyTheme(newTheme);
      localStorage.setItem('wa-theme', newTheme);
    });
  }
}

async function analyze() {
  const fileInput = document.getElementById('chatFile');
  const youName = document.getElementById('youName').value.trim();
  const friendName = document.getElementById('friendName').value.trim();
  const loading = document.getElementById('loading');
  
  if (!fileInput.files.length) {
    alert('Please upload a WhatsApp Chat (.txt) file.');
    return;
  }
  
  if (!youName || !friendName) {
    alert('Please enter both your name and your friend\'s name exactly as in the chat.');
    return;
  }
  
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('you_name', youName);
  formData.append('friend_name', friendName);

  loading.style.display = 'inline';
  document.getElementById('analyzeBtn').disabled = true;
  
  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Analysis failed');
    renderResults(data);
  } catch (err) {
    console.error('Error analyzing chat:', err);
    alert(err.message);
  } finally {
    loading.style.display = 'none';
    document.getElementById('analyzeBtn').disabled = false;
  }
}

function renderResults(data) {
  const youLabel = data.you_name || 'You';
  const otherLabel = data.friend_name || 'Friend';
  const total = data.total_messages || { you: 0, other: 0, total: 0 };
  const emojiStats = data.total_emojis || { you: 0, other: 0, total: 0 };
  const ln = data.late_night || { you: 0, other: 0 };
  const qr = data.quick_replies || { you: 0, other: 0 };
  const totalChatDays = data.total_chat_days || 0;
  const outlierCount = data.content_outlier_count || 0;
  const totalMessages = total.you + total.other;

  // Update stats cards
  document.getElementById('totalMessages').innerHTML = 
    `Total: ${totalMessages}<br>${youLabel}: ${total.you}<br>${otherLabel}: ${total.other}`;
  
  document.getElementById('emojiStats').innerHTML = 
    `Total: ${emojiStats.you + emojiStats.other}<br>${youLabel}: ${emojiStats.you}<br>${otherLabel}: ${emojiStats.other}`;
  
  const lateNightTotal = ln.you + ln.other;
  document.getElementById('lateNight').innerHTML = 
    `Total: ${lateNightTotal}<br>${youLabel}: ${ln.you}<br>${otherLabel}: ${ln.other}`;
  
  const quickRepliesTotal = qr.you + qr.other;
  document.getElementById('quickReplies').innerHTML = 
    `Total: ${quickRepliesTotal}<br>${youLabel}: ${qr.you}<br>${otherLabel}: ${qr.other}`;
  
  // Chat Timeline
  const chatTimelineEl = document.getElementById('chatTimeline');
  if (chatTimelineEl) {
    const startDate = data.chat_start || '—';
    const endDate = data.chat_end || '—';
    chatTimelineEl.innerHTML = `Start: ${startDate}<br>End: ${endDate}`;
  }

  // Outliers (system + media messages)
  const outliersEl = document.getElementById('outliers');
  if (outliersEl) {
    outliersEl.innerHTML = `Total: ${outlierCount}`;
  }

  // Render Timeline Chart
  const timelineData = data.timeline_data || [];
  if (timelineData.length > 0) {
    renderTimelineChart(timelineData);
  }
}

function renderTimelineChart(timelineData) {
  const ctx = document.getElementById('timelineChart');
  if (!ctx) return;

  // Convert dates to DD/MM/YY format for display
  const dates = timelineData.map(d => {
    const parts = d.date.split('/');
    // parts[0] = day, parts[1] = month, parts[2] = year
    return `${parts[0]}/${parts[1]}/${parts[2].slice(-2)}`;
  });
  
  const counts = timelineData.map(d => d.count);

  // Determine neon colors based on theme
  const isDark = document.body.classList.contains('dark-theme');
  // Dark mode: cyan neon (#22d3ee), Light mode: dark saturated blue (#0d47a1) for visibility
  const lineColor = isDark ? '#22d3ee' : '#0d47a1';
  const bgColor = isDark ? 'rgba(34, 211, 238, 0.15)' : 'rgba(13, 71, 161, 0.12)';
  const gridColor = isDark ? 'rgba(34, 211, 238, 0.15)' : 'rgba(13, 71, 161, 0.15)';
  const textColor = isDark ? '#e0faff' : '#020617';
  const pointColor = isDark ? '#00ffff' : '#0d47a1';
  const shadowColor = isDark ? 'rgba(34, 211, 238, 0.4)' : 'rgba(13, 71, 161, 0.3)';

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Messages per Day',
        data: counts,
        borderColor: lineColor,
        backgroundColor: bgColor,
        tension: 0.4,
        fill: true,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointBackgroundColor: pointColor,
        pointBorderColor: lineColor,
        pointBorderWidth: 2,
        borderWidth: 3,
        shadowOffsetX: 0,
        shadowOffsetY: 0,
        shadowBlur: 10,
        shadowColor: shadowColor,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { 
            color: textColor, 
            font: { size: 14, weight: 'bold' },
            usePointStyle: true,
            pointStyle: 'circle',
            padding: 20
          }
        },
        filler: {
          propagate: true
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { 
            color: textColor,
            font: { size: 12, weight: '500' }
          },
          grid: { 
            color: gridColor,
            lineWidth: 1,
            drawBorder: false
          },
          title: { 
            display: true, 
            text: 'Number of Messages',
            color: textColor,
            font: { size: 13, weight: 'bold' }
          }
        },
        x: {
          ticks: { 
            color: textColor,
            font: { size: 11, weight: '500' },
            maxRotation: 45,
            minRotation: 0
          },
          grid: { 
            color: gridColor,
            lineWidth: 0.5,
            drawBorder: false
          },
          title: { 
            display: true, 
            text: 'Date (DD/MM/YY)',
            color: textColor,
            font: { size: 13, weight: 'bold' }
          }
        }
      }
    }
  });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  document.getElementById('analyzeBtn').addEventListener('click', analyze);
});
