// CycloVision AI - Emergency Push Notification Service Worker (SIH26070)
self.addEventListener('push', function(event) {
    let data = {
        title: '🚨 CRITICAL CYCLONE ALERT | CycloVision AI',
        body: 'Very Severe Cyclone BOB-02 approaching coastal Andhra-Odisha. Landfall expected within 24h. Take immediate shelter.',
        icon: '/static/icon-192.png',
        badge: '/static/icon-badge.png',
        tag: 'cyclone-emergency-alert',
        data: { url: '/static/index.html' },
        vibrate: [300, 100, 300, 100, 300]
    };

    if (event.data) {
        try {
            data = event.data.json();
        } catch (e) {
            data.body = event.data.text();
        }
    }

    const options = {
        body: data.body,
        icon: data.icon || 'https://cdn-icons-png.flaticon.com/512/1753/1753311.png',
        badge: 'https://cdn-icons-png.flaticon.com/512/1753/1753311.png',
        vibrate: data.vibrate || [300, 100, 300, 100, 300],
        tag: data.tag || 'cyclone-emergency',
        renotify: true,
        requireInteraction: true,
        actions: [
            { action: 'shelter', title: '🏥 Find Nearest Shelter' },
            { action: 'open', title: '🌐 Open Command Center' }
        ],
        data: data.data || { url: '/static/index.html' }
    };

    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

self.addEventListener('notificationclick', function(event) {
    event.notification.close();
    if (event.action === 'shelter') {
        event.waitUntil(
            clients.openWindow('/static/index.html#shelters')
        );
    } else {
        event.waitUntil(
            clients.openWindow('/static/index.html')
        );
    }
});
