<script lang="ts">
    import type { ChatMessage } from '$lib/stores/chat';
    
    // Props
    export let onSubmit: (message: string) => Promise<void>;
    export let initialHistory: ChatMessage[] = [];  // New prop with default empty array
    
    let messageFeed: ChatMessage[] = initialHistory;
    let currentMessage = '';
    export let waitingForResponse = false;
    let chatFeedSection: HTMLElement;
    
    // Add width prop with default value
    export let width: string = "800px";
    
    // Helper function to scroll to bottom
    function scrollToBottom() {
        if (chatFeedSection) {
            chatFeedSection.scrollTop = chatFeedSection.scrollHeight;
        }
    }
    
    async function handleSubmit() {
        console.log('Send button pressed with message:', currentMessage);
        
        const messageToSend = currentMessage;
        currentMessage = '';
        
        messageFeed = [...messageFeed, {
            id: messageFeed.length,
            host: true,
            name: "You",
            timestamp: "Just now",
            message: messageToSend,
            color: "variant-soft-surface"
        }];
        
        // Add scroll to bottom after user message
        setTimeout(scrollToBottom, 50);
        
        await onSubmit(messageToSend);
    }
    
    export function addAIMessage(message: string) {
        messageFeed = [...messageFeed, {
            id: messageFeed.length,
            host: false,
            name: "Parrotfish Agent",
            timestamp: "Just now",
            message: message,
            color: "variant-soft-primary"
        }];
        
        setTimeout(scrollToBottom, 50);
    }

    // Export function to get current history
    export function getHistory(): ChatMessage[] {
        return messageFeed;
    }

    export function setWaitingForResponse(waiting: boolean) {
        waitingForResponse = waiting;
    }
</script>

<!-- Chat Card with Glow Effect -->
<div class="relative" style="width: {width}">
    <!-- Glow Background -->
    <section class="img-bg absolute z-[-1]" />
    
    <!-- Chat Interface -->
    <div class="card w-full h-[600px] grid grid-rows-[1fr_auto] z-10">
        <!-- Chat Feed -->
        <section 
            class="p-4 overflow-y-auto overflow-x-hidden space-y-4"
            bind:this={chatFeedSection}
        >
            {#each messageFeed as bubble}
                {#if bubble.host === true}
                    <div class="grid grid-cols-[1fr] gap-2">
                        <div class="card p-4 space-y-2 variant-soft-surface ml-4 mr-2">
                            <header class="flex justify-between items-center">
                                <p class="font-bold">{bubble.name}</p>
                                <small class="opacity-50">{bubble.timestamp}</small>
                            </header>
                            <p>{bubble.message}</p>
                        </div>
                    </div>
                {:else}
                    <div class="grid grid-cols-[auto_1fr] gap-2">
                        <div class="w-12 h-12 flex items-center justify-center rounded-full bg-surface-300-600-token">
                            <svg viewBox="0 0 24 24" class="w-6 h-6 fill-current">
                                <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,8.39C13.57,9.4 15.42,10 17.42,10C18.2,10 18.95,9.91 19.67,9.74C19.88,10.45 20,11.21 20,12C20,16.41 16.41,20 12,20C9,20 6.39,18.34 5,15.89L6.75,14V13A1.25,1.25 0 0,1 8,11.75A1.25,1.25 0 0,1 9.25,13V14H12M16,11.75A1.25,1.25 0 0,0 14.75,13A1.25,1.25 0 0,0 16,14.25A1.25,1.25 0 0,0 17.25,13A1.25,1.25 0 0,0 16,11.75Z" />
                            </svg>
                        </div>
                        <div class="card p-4 space-y-2 variant-soft-primary mr-8">
                            <header class="flex justify-between items-center">
                                <p class="font-bold">{bubble.name}</p>
                                <small class="opacity-50">{bubble.timestamp}</small>
                            </header>
                            <p>{bubble.message}</p>
                        </div>
                    </div>
                {/if}
            {/each}
            
            {#if waitingForResponse}
                <div class="grid grid-cols-[auto_1fr] gap-2">
                    <div class="w-12 h-12 flex items-center justify-center rounded-full bg-surface-300-600-token">
                        <svg viewBox="0 0 24 24" class="w-6 h-6 fill-current">
                            <path d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,8.39C13.57,9.4 15.42,10 17.42,10C18.2,10 18.95,9.91 19.67,9.74C19.88,10.45 20,11.21 20,12C20,16.41 16.41,20 12,20C9,20 6.39,18.34 5,15.89L6.75,14V13A1.25,1.25 0 0,1 8,11.75A1.25,1.25 0 0,1 9.25,13V14H12M16,11.75A1.25,1.25 0 0,0 14.75,13A1.25,1.25 0 0,0 16,14.25A1.25,1.25 0 0,0 17.25,13A1.25,1.25 0 0,0 16,11.75Z" />
                        </svg>
                    </div>
                    <div class="card p-4 space-y-2 variant-soft-primary mr-8">
                        <div class="flex gap-2">
                            <span class="loading loading-dots">Thinking</span>
                        </div>
                    </div>
                </div>
            {/if}
        </section>

        <!-- Input Area -->
        <section class="border-t border-surface-500/30 p-4">
            <form 
                class="flex gap-4 items-center"
                on:submit|preventDefault={handleSubmit}
            >
                <textarea
                    bind:value={currentMessage}
                    class="input bg-transparent border-2 rounded-xl p-4 flex-1 resize-none focus:border-primary-500 transition-colors"
                    placeholder="..."
                    rows="3"
                    disabled={waitingForResponse}
                ></textarea>
                <button 
                    type="submit"
                    class="btn variant-filled-primary h-[45px] px-6 rounded-xl hover:scale-105 transition-all duration-200 text-base"
                >
                    Send
                </button>
            </form>
        </section>
    </div>
</div>

<style lang="postcss">
    /* ... keep all the existing styles ... */
</style> 