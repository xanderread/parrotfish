


<svelte:head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
        crossorigin=""></script>
</svelte:head>
<script lang="ts">
  // Import Leaflet CSS and JS directly in the script

  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import ChatWidget from '$lib/components/ChatWidget.svelte';
  import flagIcon from '$lib/start.png';
  import fishIconImage from '$lib/fish.png';
  import { pageDataStore } from '$lib/stores/fishingStore';

  let chatWidget: ChatWidget;
  let waitingForResponse = false;
  let map;
  let markers = [];
  let circle;
  let dottedLine;
  let hotspots = [];
  
  // Get both fishing data and chat history from store
  let fishingData;
  let initialHistory;
  let hasInitialized = false;
  
  pageDataStore.subscribe(value => {
    fishingData = value.fishingData;
    initialHistory = value.chatHistory;
    
    // Only redirect if we've mounted and confirmed there's no data
    if (hasInitialized && !fishingData && browser) {
      goto('/');
    }
  });

  onMount(() => {
    hasInitialized = true;
    // Redirect if no fishing data is available after mount
    if (!fishingData && browser) {
      goto('/');
    }
  });

  const STARTING_ZOOM = 12;
  const FISH_PERIMETER = 1000;

  async function fetchHotspots(lat: number, lng: number, time: string, distance: number, species: string) {
    console.log("Fetching hotspots with inputs:", lat, lng, time, distance, species);
    try {
      const request = {
        lat: lat,
        lon: lng,
        max_distance: distance,
        species: species,
        time_of_fishing: time
      };
      
      const response = await fetch('http://localhost:8000/trident', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request)
      });
      
      console.log("Sending to trident:", JSON.stringify(request));

      if (!response.ok) {
        throw new Error('Failed to fetch hotspots');
      }

      const data = await response.json();
      console.log("Received hotspots:", JSON.stringify(data.hotspots));
      return data.hotspots[0]; // Return first hotspot
    } catch (error) {
      console.error('Error fetching hotspots:', error);
      return null;
    }
  }

  onMount(async () => {
    if (browser && fishingData) {
      // Use data from store instead of hardcoded values
      let lat = fishingData.location[0].latitude;
      let lon = fishingData.location[0].longitude;
      let time = fishingData.time;
      console.log("I am using a time of this from zzee output " + time);
      let distance = fishingData.distance;
      let species = fishingData.species[0];

      console.log("I am using lat and lng of " + lat + " and " + lon);
      
      // Fetch hotspots and get best location
      console.log("I am fetching hotspots", lat, lon, time, distance, species);
      const bestHotspot = await fetchHotspots(lat, lon, time, distance, species);

      
      
      // Use the best hotspot coordinates or fallback
      const FISH_LAT = bestHotspot?.lat 
      const FISH_LNG = bestHotspot?.long 
      const FISH_ZOOM = 12;
      const FISH_PERIMETER = 1000; // Convert km to meters

      // Initialize map
      map = L.map('map');
      
      var startingIcon = L.icon({
    iconUrl: flagIcon,
    iconSize:     [64,64], // size of the icon
        iconAnchor:  [32, 50], // point of the icon which will correspond to marker's location
      });
      L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.jpg', {
        minZoom: 0,
        maxZoom: 20,
        attribution: '&copy; CNES, Distribution Airbus DS, © Airbus DS, © PlanetObserver (Contains Copernicus Data) | &copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(map);

      // Add a marker at the starting position
      L.marker([fishingData?.location[0]?.latitude, fishingData?.location[0]?.longitude], {icon: startingIcon})
        .addTo(map)
        .bindPopup(`
          <div style="background: rgb(35, 46, 67); color: rgb(198, 204, 215); padding: 1rem; border-radius: 8px;">
            <div style="font-size: 1.2rem; margin-bottom: 1rem;">Starting Location</div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
              <span>Latitude</span>
              <span>${fishingData?.location[0]?.latitude.toFixed(4)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
              <span>Longitude</span>
              <span>${fishingData?.location[0]?.longitude.toFixed(4)}</span>
            </div>
          </div>
        `)
        

      var fishMarkerIcon = L.icon({
        iconUrl: fishIconImage,
        iconSize:     [64,64], // size of the icon
        iconAnchor:  [32, 32], // point of the icon which will correspond to marker's location
      });

      L.marker([FISH_LAT, FISH_LNG], {icon: fishMarkerIcon})
        .addTo(map)
        .bindPopup(`
          <div style="background: rgb(35, 46, 67); color: rgb(198, 204, 215); padding: 1rem; border-radius: 0.5rem;">
            <div style="font-size: 1rem; margin-bottom: 0.75rem; font-weight: bold;">Recommended Location</div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 0.875rem;">
              <span>Latitude</span>
              <span>${FISH_LAT?.toFixed(4) || 'Unavailable'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 0.875rem;">
              <span>Longitude</span>
              <span>${FISH_LNG?.toFixed(4) || 'Unavailable'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 0.875rem;">
              <span>Species</span>
              <span>${species?.charAt(0).toUpperCase() + species?.slice(1) || 'Unavailable'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 0.875rem;">
              <span>Fish Density Score</span>
              <span>${bestHotspot?.fish_population_score.toFixed(2) || 'Unavailable'}</span>
            </div>
          </div>
        `, {
          offset: [0, -80], // Offset to position above the marker
          className: 'custom-popup'
        })
        .openPopup()
        .on('click', () => {
          map.setView([FISH_LAT, FISH_LNG], FISH_ZOOM + 3, {
            animate: true,
            duration: 1
          });
        });

      // Add circle around fish location with fixed meter radius
      L.circle([FISH_LAT, FISH_LNG], {
        radius: FISH_PERIMETER,
        color: 'rgb(198, 204, 215)',
        fillColor: 'red',
        fillOpacity: 0.2,
        weight: 1,
        dashArray: '10, 10'
      }).addTo(map);

      // Add the animated dotted line between points
      const pathCoordinates = [
        [fishingData?.location[0]?.latitude, fishingData?.location[0]?.longitude],
        [FISH_LAT, FISH_LNG]
      ];
      
      const dottedLine = L.polyline(pathCoordinates, {
        color: 'rgb(198, 204, 215)',
        dashArray: '10, 10',
        weight: 1,
        opacity: 0.8,
        className: 'animated-line'
      }).addTo(map);

      // Create an array of points we want to include in view
      const points = [
        [fishingData?.location[0]?.latitude, fishingData?.location[0]?.longitude],
        [FISH_LAT, FISH_LNG]
      ];

      // Create a bounds object
      const bounds = L.latLngBounds(points);

      // Fit the map to show these bounds with some padding
      map.fitBounds(bounds, {
        padding: [50, 50], // [top/bottom, left/right] padding in pixels
        maxZoom: 15        // Prevent it from zooming in too far
      });

      // Add initial AI messages only if there's no history
      if (initialHistory.length === 0) {
        chatWidget.addAIMessage("I can help you explore this area. What would you like to know?");
        chatWidget.addAIMessage("I'm here to help you with your fishing plans.");
      }
    }
  });

  async function handleChatSubmit(message: string) {
    try {
        // Set loading state
        chatWidget.setWaitingForResponse(true);
        waitingForResponse = true;
        
        // Add user's message to the chat
        const messageToSend = message.trim();
        if (!messageToSend) return;
        
        console.log("I am using a chat session id of " + localStorage.getItem('chat_session_id'));
        
        // Make API call to backend
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: messageToSend,
                session_id: localStorage.getItem('chat_session_id') || null
            })
        });




        if (!response.ok) {
            throw new Error('Failed to get response from server');
        }

        const data = await response.json();

        localStorage.setItem('chat_session_id', data.session_id);



        if (data.fishing_plan && Object.keys(data.fishing_plan).length > 0) {
            // Extract the JSON data from extractedInformation section
            const extractedInfoMatch = data.message.match(/<extractedInformation>(.*?)<\/extractedInformation>/s);
            if (extractedInfoMatch) {
                try {
                    console.log("I am parsing the fishing info");
                    const fishingInfo = JSON.parse(extractedInfoMatch[1]);
                    
                    // Convert the location array format to match expected structure
                    const locationObj = {
                        location: [{
                            latitude: fishingInfo.location[0],
                            longitude: fishingInfo.location[1]
                        }],
                        species: fishingInfo.species,
                        time: fishingInfo.time,
                        distance: fishingInfo.distance
                    };


                    // Get hotspots from Trident API
                    const tridentHotspot = await fetchHotspots(
                        locationObj.location[0].latitude,
                        locationObj.location[0].longitude,
                        locationObj.time,
                        locationObj.distance,
                        locationObj.species[0]
                    );
                    
                    console.log("if i appear before Im trouble");
                
                    if (tridentHotspot) {
                        // Update map with new coordinates
                        updateMapCoordinates(
                            locationObj.location[0].latitude,
                            locationObj.location[0].longitude,
                            tridentHotspot.lat,
                            tridentHotspot.long
                        );
                    } else {
                      
                      alert("Could not find good fishing spots. Please try again.");
                    }
                } catch (e) {
                    console.error('Failed to parse fishing info:', e);
                }
            }
        }

       

        chatWidget.addAIMessage(data.message);
        
    } catch (error) {
        console.error('Error:', error);
        chatWidget.addAIMessage("Sorry, I encountered an error. Please try again.");
    } finally {
        waitingForResponse = false;
    }
  }

  // Function to update map coordinates
  function updateMapCoordinates(newStartLat, newStartLng, newFishLat, newFishLng) {
    if (!map) return;
    
    // Remove the old map completely
    map.remove();
    
    // Create new map instance
    map = L.map('map');
    
    var startingIcon = L.icon({
      iconUrl: flagIcon,
      iconSize:     [64,64], // size of the icon
      iconAnchor:  [32, 32], // point of the icon which will correspond to marker's location
    });
    L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.jpg', {
      minZoom: 0,
      maxZoom: 20,
      attribution: '&copy; CNES, Distribution Airbus DS, © Airbus DS, © PlanetObserver (Contains Copernicus Data) | &copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Add a marker at the starting position
    L.marker([newStartLat, newStartLng], {icon: startingIcon})
      .addTo(map)
      

    var fishMarkerIcon = L.icon({
      iconUrl: fishIconImage,
      iconSize:     [64,64], // size of the icon
      iconAnchor:  [32, 32], // point of the icon which will correspond to marker's location
    });

    L.marker([newFishLat, newFishLng], {icon: fishMarkerIcon})
      .addTo(map)
      .bindPopup(`
        <div style="background: rgb(35, 46, 67); color: rgb(198, 204, 215); padding: 1rem; border-radius: 8px;">
          <div style="font-size: 1.2rem; margin-bottom: 1rem;">Fish Location</div>
          <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
            <span>Location</span>
            <span>Miami, FL</span>
          </div>
        </div>
      `)
      .openPopup()
      .on('click', () => {
        map.setView([newFishLat, newFishLng], 15, {  // Using fixed zoom level of 15
          animate: true,
          duration: 1
        });
      });

    // Add circle around fish location with fixed meter radius
    L.circle([newFishLat, newFishLng], {
      radius: FISH_PERIMETER,
      color: 'rgb(198, 204, 215)',
      fillColor: 'red',
      fillOpacity: 0.2,
      weight: 1,
      dashArray: '10, 10'
    }).addTo(map);

    // Add the animated dotted line between points
    const pathCoordinates = [
      [newStartLat, newStartLng],
      [newFishLat, newFishLng]
    ];
    
    const dottedLine = L.polyline(pathCoordinates, {
      color: 'rgb(198, 204, 215)',
      dashArray: '10, 10',
      weight: 1,
      opacity: 0.8,
      className: 'animated-line'
    }).addTo(map);

    // Create an array of points we want to include in view
    const points = [
      [newStartLat, newStartLng],
      [newFishLat, newFishLng]
    ];

    // Create a bounds object
    const bounds = L.latLngBounds(points);

    // Fit the map to show these bounds with some padding
    map.fitBounds(bounds, {
      padding: [50, 50], // [top/bottom, left/right] padding in pixels
      maxZoom: 15        // Prevent it from zooming in too far
    });
  }
</script>

<div class="layout-container">
  <div id="map"></div>
  <div class="chat-section">
    <ChatWidget 
      bind:this={chatWidget}
      onSubmit={handleChatSubmit} 
      initialHistory={initialHistory}
      bind:waitingForResponse
      width="500px"
    />
  </div>
</div>

<style>
  .layout-container {
    display: flex;
    width: 100%;
    height: 100vh;
    overflow: hidden;
  }

  #map {
    flex: 2;
    height: 100vh;
    width: 100%;
  }

  .chat-section {
    flex: 1;
    height: 100vh;
  }

  /* Existing chat section styles */
  :global(.chat-section .card) {
    height: 100% !important;
    width: 100% !important;
    border-radius: 0;
  }

  :global(.chat-section > div) {
    height: 100% !important;
  }

  /* Add the animation for the dotted line */
  :global(.animated-line) {
    stroke-dashoffset: 20;
    animation: dashdraw 2s linear infinite;
  }

  @keyframes dashdraw {
    from {
      stroke-dashoffset: 40;
    }
    to {
      stroke-dashoffset: 0;
    }
  }

  /* Update Leaflet popup styles */
  :global(.leaflet-popup-content-wrapper),
  :global(.leaflet-popup) {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
  }

  :global(.leaflet-container) {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  }

  :global(.leaflet-popup-content) {
    margin: 0 !important;
    min-width: 400px;
  }

  :global(.leaflet-popup-tip),
  :global(.leaflet-popup-tip-container) {
    display: none !important;
  }

  :global(.leaflet-popup-close-button) {
    color: rgb(198, 204, 215);
    margin: 4px;
  }

  :global(.custom-popup .leaflet-popup-content-wrapper),
  :global(.custom-popup.leaflet-popup) {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
  }

  :global(.custom-popup .leaflet-popup-content) {
    margin: 0;
  }
</style>


