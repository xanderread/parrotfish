import { writable } from 'svelte/store';
import type { ChatMessage } from '$lib/stores/chat';

interface FishingData {
    location: Array<{
        latitude: number;
        longitude: number;
    }>;
    species: string[];
    time: string;
    distance: number;
}

interface PageData {
    fishingData: FishingData | null;
    chatHistory: ChatMessage[];
}

export const pageDataStore = writable<PageData>({
    fishingData: null,
    chatHistory: []
});