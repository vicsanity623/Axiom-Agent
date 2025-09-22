const CACHE_NAME = 'axiom-agent-cache-v0.0.2'; // Increment version to force update
const URLS_TO_CACHE = [
  '/',
  '/static/style.css',
  '/manifest.json', // It's better to cache the manifest itself
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// Install: Caches the app shell
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache and caching app shell');
        return cache.addAll(URLS_TO_CACHE);
      })
  );
});

// Activate: Cleans up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: clearing old cache');
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Fetch: Implements a "network falling back to cache" strategy
self.addEventListener('fetch', event => {
  // Always go to the network for API calls
  if (event.request.url.includes('/chat') || event.request.url.includes('/status')) {
    return event.respondWith(fetch(event.request));
  }

  event.respondWith(
    fetch(event.request).catch(() => {
      // If the network request fails, try to find it in the cache
      return caches.match(event.request).then(response => {
        if (response) {
          return response;
        }
        // If it's not in the cache either, well, we can't do anything.
        // This is where you might show a custom offline page in a more complex app.
      });
    })
  );
});
