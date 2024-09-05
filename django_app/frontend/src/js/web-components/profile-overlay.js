// @ts-check

class ProfileOverlay extends HTMLElement {
  constructor() {
    super();
    this.currentPage = 0;
  }
  connectedCallback() {
    const dialog = this.querySelector("dialog");
    if (!dialog) {
      return;
    }

    // display dialog on page load?
    if (this.dataset.show === "true") {
      dialog.showModal();
    }

    const changePage = (newPageIndex) => {
      let pages = this.querySelectorAll(".profile-page");
      pages.forEach((page, pageIndex) => {
        if (pageIndex === newPageIndex) {
          page.classList.remove("profile--hidden");
          /** @type {HTMLInputElement | null} */
          const firstInteractiveElement = page.querySelector(
            "input, textarea, .profile__button-finish"
          );
          firstInteractiveElement?.focus();
        } else {
          page.classList.add("profile--hidden");
        }
      });
    };
    changePage(this.currentPage);

    let nextButtons = this.querySelectorAll(`button[data-action="next"]`);
    nextButtons.forEach((nextButton) => {
      nextButton.addEventListener("click", () => {
        let checksPassed = true;
        /** @type {NodeListOf<HTMLInputElement> | undefined} */
        let inputs = nextButton
          .closest(".profile-page")
          ?.querySelectorAll("input, textarea");
        inputs?.forEach((input) => {
          if (!input.checkValidity()) {
            input.reportValidity();
            checksPassed = false;
          }
        });
        if (checksPassed) {
          this.currentPage++;
          changePage(this.currentPage);
        }
      });
    });

    let previousButtons = this.querySelectorAll(
      `button[data-action="previous"]`
    );
    previousButtons.forEach((previousButton) => {
      previousButton.addEventListener("click", () => {
        this.currentPage--;
        changePage(this.currentPage);
      });
    });

    let skipButtons = this.querySelectorAll(`button[data-action="skip"]`);
    skipButtons.forEach((skipButton) => {
      skipButton.addEventListener("click", () => {
        this.currentPage++;
        changePage(this.currentPage);
      });
    });

    let closeButtons = this.querySelectorAll(`button[data-action="close"]`);
    closeButtons.forEach((closeButton) => {
      closeButton.addEventListener("click", () => {
        this.currentPage = 0;
        this.querySelector("dialog")?.close();

        /**
         * Sends data to endpoint, retrying if necessary
         * @param {string} data
         * @param {number} retry
         */
        const sendData = async (data, retry = 0) => {
          const MAX_RETRIES = 10;
          const RETRY_INTERVAL_SECONDS = 10;
          const csrfToken =
            /** @type {HTMLInputElement | null} */ (
              this.querySelector('[name="csrfmiddlewaretoken"]')
            )?.value || "";
          try {
            await fetch(`/update-profile`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": csrfToken,
              },
              body: JSON.stringify(data),
            });
          } catch (err) {
            if (retry < MAX_RETRIES) {
              window.setTimeout(() => {
                sendData(data, retry + 1);
              }, RETRY_INTERVAL_SECONDS * 1000);
            }
          }
        };

        // Format and send data
        const formData = new FormData(this.querySelector("form") || undefined);
        let formDataJson = {};
        for (const pair of formData.entries()) {
          formDataJson[pair[0]] = pair[1];
        }
        sendData(formDataJson.toString());
      });
    });
  }
}
customElements.define("profile-overlay", ProfileOverlay);
