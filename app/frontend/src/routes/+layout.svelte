<script lang="ts">
	import '../app.postcss';
	import { AppShell, AppBar } from '@skeletonlabs/skeleton';
	import { initializeStores, Modal } from '@skeletonlabs/skeleton';
	import { computePosition, autoUpdate, flip, shift, offset, arrow } from '@floating-ui/dom';
	import { storePopup } from '@skeletonlabs/skeleton';
	import { fade } from 'svelte/transition';
	import { page } from '$app/stores';

	initializeStores();
	
	// Floating UI for Popups - only need to set this once
	storePopup.set({ computePosition, autoUpdate, flip, shift, offset, arrow });
</script>

<Modal />
<!-- App Shell -->
<AppShell>
	<slot />
</AppShell>
