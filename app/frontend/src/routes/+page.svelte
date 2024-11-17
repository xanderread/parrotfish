<script lang="ts">
    import ChatWidget from '$lib/components/ChatWidget.svelte';
    import { pageDataStore } from '$lib/stores/fishingStore';
    import { goto } from '$app/navigation';
    let videoElement: HTMLVideoElement;
    let chatWidget: ChatWidget;
    let loading = false;
    
    // Set playback rate when video element is mounted
    $: if (videoElement) {
        videoElement.playbackRate = 0.7;
    }
    
    // Add initial welcome message when component mounts
    $: if (chatWidget) {
        chatWidget.addAIMessage("Hello! I'd love to help plan your fishing expedition. Tell me about where you'd like to fish, what kind of fish you're looking for and how far you're willing to travel.");
    }
    
    async function handleChatSubmit(message: string) {
        try {
            // Set loading state to true before making the request
            chatWidget.setWaitingForResponse(true);
            loading = true;
            
            // Make API call to backend
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: localStorage.getItem('chat_session_id') || null
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get response from server');
            }

            const data = await response.json();
            
            // Store the session ID for future requests
            localStorage.setItem('chat_session_id', data.session_id);
            // Add AI response to chat
            // Extract message between <reply> tags and remove <thought> and <extractedInformation> sections
            const replyMatch = data.message.match(/<reply>(.*?)<\/reply>/s);
            
            let formattedMessage;
            if (replyMatch && replyMatch[1]) {
                formattedMessage = replyMatch[1].trim();
            } else {
                // If no reply tags or match is null, use the original message
                formattedMessage = "Awesome! Let's plan your fishing trip.";
            }
            
            chatWidget.addAIMessage(formattedMessage);
        

            // If fishing plan is available and complete, redirect to map
            // Change 
            if (data.fishing_plan && Object.keys(data.fishing_plan).length > 0) {
                const extractedInfoMatch = data.message.match(/<extractedInformation>(.*?)<\/extractedInformation>/s);
                if (extractedInfoMatch) {
                    const fishingInfo = JSON.parse(extractedInfoMatch[1]);
                    
                    // Update the store with fishing data and chat history
                    pageDataStore.update(currentData => ({
                        fishingData: {
                            location: [{
                                latitude: fishingInfo.location[0],
                                longitude: fishingInfo.location[1]
                            }],
                            species: fishingInfo.species,
                            time: fishingInfo.time,
                            distance: fishingInfo.distance
                        },
                        chatHistory: chatWidget.getHistory()
                    }));

                    // Wait for store update to complete
                    await new Promise(resolve => setTimeout(resolve, 100));
                    
                    // Use goto instead of window.location
                    await goto('/map');
                }
            }
            
        } catch (error) {
            console.error('Error:', error);
            chatWidget.addAIMessage("Sorry, I encountered an error. Please try again.");
        } finally {
            loading = false;
            chatWidget.setWaitingForResponse(false);
        }
    }
</script>

<div class="h-screen overflow-hidden flex justify-center items-center">
    <!-- Video Background -->
    <video
        class="fixed inset-0 w-full h-full object-cover -z-10 opacity-20"
        autoplay
        muted
        loop
        playsinline
        bind:this={videoElement}
    >
        <source src="/videos/ocean.mp4" type="video/mp4" />
        Your browser does not support the video tag.
    </video>

    <div class="space-y-8 flex flex-col items-center">
        <!-- Header -->
        <div class="flex flex-col items-center gap-2">
            <h1 class="h1 hover-expand">Parrotfish.ai</h1>
            <p><u><b>Plan your fishing expedition in minutes, not months</b></u></p>
        </div>

        <!-- Added animated glow container -->
        <figure class="relative px-1 md:px-2 w-full max-w-1xl">
            <div class="img-bg"></div>
            <ChatWidget 
                bind:this={chatWidget}
                onSubmit={handleChatSubmit}
            />
        </figure>
    </div>
</div>

<style lang="postcss">
    :global(body) {
        background-color: rgba(0, 0, 0, 0.4);
    }
    :global(html, body) {
        @apply h-full overflow-hidden;
    }

    figure {
        @apply flex relative;
    }
    
    .img-bg {
        @apply absolute z-[-1] rounded-full blur-[50px] transition-all;
        width: 100%;
        height: 100%;
        animation: pulse 5s cubic-bezier(0, 0, 0, 0.5) infinite,
            glow 5s linear infinite;
    }

    @keyframes glow {
        0% {
            @apply bg-primary-400/50;
        }
        33% {
            @apply bg-secondary-400/50;
        }
        66% {
            @apply bg-tertiary-400/50;
        }
        100% {
            @apply bg-primary-400/50;
        }
    }

    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
        }
    }

    .hover-expand {
        transition: transform 0.3s ease;
    }

    .hover-expand:hover {
        transform: scale(1.1);
    }
</style>
