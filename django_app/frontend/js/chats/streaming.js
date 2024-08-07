// @ts-check

class SourcesList extends HTMLElement {

    constructor() {
        super();
        this.sources = [];
    }

    /**
     * Adds a source to the current message
     * @param {string} fileName 
     * @param {string} url 
     */
    add = (fileName, url) => {
        
        this.sources.push({
            fileName: fileName,
            url: url
        });

        let html = `
            <h3 class="iai-chat-bubble__sources-heading govuk-heading-s govuk-!-margin-bottom-1">Sources</h3>
            <div class="iai-display-flex-from-desktop">
            <ul class="govuk-list govuk-list--bullet govuk-!-margin-bottom-0">
        `;
        this.sources.forEach((source) => {
            html += `
                <li class="govuk-!-margin-bottom-0">
                    <a class="iai-chat-bubbles__sources-link govuk-link" href="${source.url}">${source.fileName}</a>
                </li>
            `;
        });
        html += `</div></ul>`;
    
        this.innerHTML = html;

    }


    /** 
     * Shows to citations link/button
     * @param {string} chatId
     */
    showCitations = (chatId) => {
        if (!chatId) {
            return;
        }
        let link = document.createElement('a');
        link.classList.add('iai-chat-bubble__citations-button', 'govuk-!-margin-left-2');
        link.setAttribute('href', `/citations/${chatId}`);
        link.textContent = 'View information behind this answer';
        this.querySelector('.iai-display-flex-from-desktop')?.appendChild(link);
    }

}
customElements.define('sources-list', SourcesList);




class ChatMessage extends HTMLElement {

    connectedCallback() {
        this.innerHTML = `
            <div class="iai-chat-bubble iai-chat-bubble--${this.dataset.role === 'user' ? 'right' : 'left'} js-chat-message govuk-body {{ classes }}" data-role="{{ role }}" tabindex="-1">
                <div class="iai-chat-bubble__role">${this.dataset.role === 'ai' ? 'Redbox' : 'You'}</div>
                <div class="iai-chat-bubble__route" hidden></div>
                <markdown-converter class="iai-chat-bubble__text">${this.dataset.text || ''}</markdown-converter>
                ${!this.dataset.text ?
                    `<div class="rb-loading-ellipsis govuk-body-s">
                        Loading
                        <span aria-hidden="true">.</span>
                        <span aria-hidden="true">.</span>
                        <span aria-hidden="true">.</span>
                    </div>`
                : ''}
                <sources-list></sources-list>
                <div class="govuk-error-summary" data-module="govuk-error-summary" hidden>
                  <div role="alert">
                    <h2 class="govuk-error-summary__title">Error</h2>
                    <div class="govuk-error-summary__body">
                      <ul class="govuk-list govuk-error-summary__list">
                        <li>There was an unexpected error communicating with Redbox. Please try again, and contact <a href="/support/">support</a> if the problem persists.</li>
                      </ul>
                    </div>
                  </div>
                </div>
            </div>
        `;
    }

    /**
     * Streams an LLM response
     * @param {string} message
     * @param {string[]} selectedDocuments
     * @param {string | undefined} sessionId
     * @param {string} endPoint
     * @param {HTMLElement} chatControllerRef
     */
    stream = (message, selectedDocuments, sessionId, endPoint, chatControllerRef) => {

        let responseContainer = /** @type MarkdownConverter */(this.querySelector('markdown-converter'));
        let sourcesContainer = /** @type SourcesList */(this.querySelector('sources-list'));
        let responseLoading = /** @type HTMLElement */(this.querySelector('.rb-loading-ellipsis'));
        let webSocket = new WebSocket(endPoint);
        let streamedContent = '';
        let sources = [];

        // Stop streaming on escape key press
        this.addEventListener('keydown', (evt) => {
            if (evt.key === 'Escape' && this.dataset.status === 'streaming') {
                this.dataset.status = 'stopped';
                webSocket.close();
            }
        });
    
        webSocket.onopen = (event) => {
            webSocket.send(JSON.stringify({message: message, sessionId: sessionId, selectedFiles: selectedDocuments}));
            this.dataset.status = 'streaming';
        };
    
        webSocket.onerror = (event) => {
            responseContainer.innerHTML = 'There was a problem. Please try sending this message again.';
            this.dataset.status = 'error';
        };

        webSocket.onclose = (event) => {
            responseLoading.style.display = 'none';
            if (this.dataset.status !== 'stopped') {
                this.dataset.status = 'complete';
            }
        };
    
        webSocket.onmessage = (event) => {
            
            let message;
            try {
                message = JSON.parse(event.data);
            } catch(err) {
                console.log('Error getting JSON response', err);
            }

            if (message.type === 'text') {
                streamedContent += message.data;
                responseContainer.update(streamedContent);
            } else if (message.type === 'session-id') {
                chatControllerRef.dataset.sessionId = message.data;
            } else if (message.type === 'source') {
                sourcesContainer.add(message.data.original_file_name, message.data.url);
            } else if (message.type === 'route') {
                let route = this.querySelector(".iai-chat-bubble__route");
                if (route) {
                    route.textContent = message.data;
                    route.removeAttribute("hidden");
                }

                // send route to Plausible
                let plausible = /** @type {any} */ (window).plausible;
                if (typeof(plausible) !== 'undefined') {
                    plausible('Chat-message-route', {props: {route: message.data}});
                }
            } else if (message.type === 'hidden-route') {
                // TODO(@rachaelcodes): remove hidden-route with new route design
                // https://technologyprogramme.atlassian.net/browse/REDBOX-419

                // send route to Plausible
                let plausible = /** @type {any} */ (window).plausible;
                if (typeof(plausible) !== 'undefined') {
                    plausible('Chat-message-route', {props: {route: message.data}});
                }
            } else if (message.type === 'end') {
                sourcesContainer.showCitations(message.data.message_id);
            } else if (message.type === 'error') {
                this.querySelector(".govuk-error-summary")?.removeAttribute("hidden");
            }
            
        };
    
    };

}
customElements.define('chat-message', ChatMessage);




class ChatController extends HTMLElement {

    connectedCallback() {

        const messageForm = this.closest('form');
        const textArea = /** @type {HTMLInputElement | null} */ (this.querySelector('.js-user-text'));
        const messageContainer = this.querySelector('.js-message-container');
        const insertPosition = this.querySelector('.js-response-feedback');
        const feedbackButtons = /** @type {HTMLElement | null} */ (this.querySelector('feedback-buttons'));
        let selectedDocuments = [];

        messageForm?.addEventListener('submit', (evt) => {
            
            evt.preventDefault();
            const userText = textArea?.value.trim();
            if (!textArea || !userText) {
                return;
            }

            let userMessage = document.createElement('chat-message');
            userMessage.setAttribute('data-text', userText);
            userMessage.setAttribute('data-role', 'user');
            messageContainer?.insertBefore(userMessage, insertPosition);

            let aiMessage = /** @type {ChatMessage} */ (document.createElement('chat-message'));
            aiMessage.setAttribute('data-role', 'ai');
            aiMessage.setAttribute('tabindex', '-1');
            messageContainer?.insertBefore(aiMessage, insertPosition);
            aiMessage.stream(userText, selectedDocuments, this.dataset.sessionId, this.dataset.streamUrl || '', this);
            aiMessage.focus();

            // reset UI 
            if (feedbackButtons) {
                feedbackButtons.dataset.status = "";
            }
            textArea.value = "";

        });

        document.body.addEventListener('selected-docs-change', (evt) => {
            selectedDocuments = /** @type{CustomEvent} */(evt).detail;
        });

    }
  
}
customElements.define('chat-controller', ChatController);




class DocumentSelector extends HTMLElement {

    connectedCallback() {

        const documents = /** @type {NodeListOf<HTMLInputElement>} */ (this.querySelectorAll('input[type="checkbox"]'));

        const getSelectedDocuments = () => {
            let selectedDocuments = [];
            documents.forEach((document) => {
                if (document.checked) {
                    selectedDocuments.push(document.value);
                }
            });
            const evt = new CustomEvent('selected-docs-change', { detail: selectedDocuments });
            document.body.dispatchEvent(evt);
        }

        // update on page load
        getSelectedDocuments();

        // update on any selection change
        documents.forEach((document) => {
            document.addEventListener('change', getSelectedDocuments);
        });

    }
  
}
customElements.define('document-selector', DocumentSelector);
