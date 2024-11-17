import type { PageLoad } from './$types';

export const load: PageLoad = async ({ parent }) => {
    // You can get data from parent layouts or other sources
    const parentData = await parent();
    
    return {
        fishingData: parentData.fishingData
    };
}; 