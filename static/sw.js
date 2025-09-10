// static/sw.js

const CACHE_NAME = 'axiom-agent-cache-v1';
const URLS_TO_CACHE = [
  '/', // This caches the index.html at the root
  '/static/style.css',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// Install event: opens a cache and adds our essential files to it.
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        console.log('Opened cache');
        return cache.addAll(URLS_TO_CACHE);
      })
  );
});

// Fetch event: serves assets from the cache first for speed and offline capability.
self.addEventListener('fetch', function(event) {
  // We only want to cache GET requests for our app shell
  if (event.request.method !== 'GET') {
    return;
  }

  // For API calls (like /chat or /status), always go to the network.
  if (event.request.url.includes('/chat') || event.request.url.includes('/status')) {
    event.respondWith(fetch(event.request));
    return;
  }

  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        // Cache hit - return response from the cache
        if (response) {
          return response;
        }
        // Not in cache - fetch from network
        return fetch(event.request);
      }
    )
  );
});