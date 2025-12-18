const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
const resultsHeader = document.getElementById('resultsHeader');
const resultCount = document.getElementById('resultCount');

searchBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    if (!query) return;

    // UI Reset
    resultsDiv.innerHTML = '';
    loadingDiv.classList.remove('hidden');
    resultsHeader.classList.add('hidden');
    searchBtn.disabled = true;

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            throw new Error('API Error');
        }

        const data = await response.json();
        renderResults(data);

    } catch (error) {
        console.error(error);
        resultsDiv.innerHTML = '<p class="error">Something went wrong. Please try again.</p>';
    } finally {
        loadingDiv.classList.add('hidden');
        searchBtn.disabled = false;
    }
});

function renderResults(items) {
    if (!items || items.length === 0) {
        resultsDiv.innerHTML = '<p>No recommendations found.</p>';
        return;
    }

    resultsHeader.classList.remove('hidden');
    resultCount.textContent = `Found ${items.length} relevant assessments based on your requirements.`;

    items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'card';

        // Check types for badges
        // item['Type'] is a list of strings
        let badgesHtml = '';
        if (item.Type && Array.isArray(item.Type)) {
            item.Type.forEach(type => {
                const isSoft = type.includes('Personality') || type.includes('Behavior') || type.includes('Competenc');
                badgesHtml += `<span class="badge ${isSoft ? 'soft' : ''}">${type}</span>`;
            });
        }

        const scorePct = Math.round(item.Score * 100);

        card.innerHTML = `
            <div>
                <h3>${item['Assessment Name']}</h3>
                <div class="badges">
                    ${badgesHtml}
                </div>
            </div>
            <div>
                <div class="score">Relevance Score: ${scorePct}%</div>
                <a href="${item.URL}" target="_blank">View details &rarr;</a>
            </div>
        `;
        resultsDiv.appendChild(card);
    });
}
