import type { Actions } from '@sveltejs/kit';

export const actions: Actions = {
    chat: async ({ request }) => {
        try {
            const { message, session_id } = await request.json();
            console.log(message, session_id);
            const response = await fetch('http://127.0.0.1:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message, session_id })
            });

            if (!response.ok) {
                return { status: 400, error: 'Failed to get response from server' };
            }

            const data = await response.json();
            return { success: true, data };
            
        } catch (error) {
            console.error('Error:', error);
            return { 
                status: 500, 
                error: 'Internal server error' 
            };
        }
    }
}; 