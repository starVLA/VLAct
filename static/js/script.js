const VIDEO_ROOT = "./static/videos/";

const tasks = [
  {
    id: "press-button",
    orientation: "portrait",
    name: "Press Button",
    platform: "Single Arm",
    inference: `${VIDEO_ROOT}press-button.mp4`,
  },
  {
    id: "stack-blocks",
    orientation: "portrait",
    name: "Stack Blocks",
    platform: "Single Arm",
    inference: `${VIDEO_ROOT}stack-blocks.mp4`,
  },
  {
    id: "clean-table",
    orientation: "landscape",
    name: "Clean Table",
    platform: "Single Arm",
    inference: `${VIDEO_ROOT}clean-table.mp4`,
  },
  {
    id: "take-carrot-from-pot",
    orientation: "landscape",
    name: "Take Carrot From Pot",
    platform: "Single Arm",
    inference: `${VIDEO_ROOT}take-carrot-from-pot.mp4`,
  },
  {
    id: "place-pen-into-cup",
    orientation: "portrait",
    name: "Place Pen Into Cup",
    platform: "Single Arm",
    inference: `${VIDEO_ROOT}place-pen-into-cup.mp4`,
  },
  {
    id: "scoop-beans",
    orientation: "landscape",
    name: "Scoop Beans",
    platform: "Single Arm",
    inference: `${VIDEO_ROOT}scoop-beans.mp4`,
  },
  {
    id: "fold-towel",
    orientation: "landscape",
    name: "Fold Towel",
    platform: "Dual Arm",
    inference: `${VIDEO_ROOT}fold-towel.mp4`,
  },
  {
    id: "fold-pants",
    orientation: "landscape",
    name: "Fold Pants",
    platform: "Dual Arm",
    inference: `${VIDEO_ROOT}fold-pants.mp4`,
  },
  {
    id: "handover-banana",
    orientation: "landscape",
    name: "Handover Banana",
    platform: "Dual Arm",
    inference: `${VIDEO_ROOT}handover-banana.mp4`,
  },
  {
    id: "prepare-breakfast",
    orientation: "landscape",
    name: "Prepare Breakfast",
    platform: "Dual Arm",
    inference: `${VIDEO_ROOT}prepare-breakfast.mp4`,
  },
  {
    id: "unplug-cable",
    orientation: "landscape",
    name: "Unplug Cable",
    platform: "Dual Arm",
    inference: `${VIDEO_ROOT}unplug-cable.mp4`,
  },
];

const extensionToMime = {
  m4v: "video/mp4",
  mov: "video/quicktime",
  mp4: "video/mp4",
  webm: "video/webm",
};

function getMimeType(src) {
  const extension = src.split(".").pop().toLowerCase();
  return extensionToMime[extension] || "video/mp4";
}

function enforceSilentVideo(video) {
  video.muted = true;
  video.defaultMuted = true;
  video.volume = 0;
}

function createSilentVideo(src, label, className = "") {
  const video = document.createElement("video");
  video.className = className;
  video.controls = false;
  video.disablePictureInPicture = true;
  video.loop = true;
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.volume = 0;
  video.setAttribute("muted", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("aria-label", label);
  video.setAttribute("controlslist", "nodownload noplaybackrate noremoteplayback");

  const source = document.createElement("source");
  source.src = encodeURI(src);
  source.type = getMimeType(src);
  video.append(source);

  ["play", "playing", "volumechange", "loadedmetadata"].forEach((eventName) => {
    video.addEventListener(eventName, () => enforceSilentVideo(video));
  });

  return video;
}

function attachPlayerControls(video, stage, label) {
  const progress = document.createElement("input");
  progress.className = "video-progress";
  progress.type = "range";
  progress.min = "0";
  progress.max = "100";
  progress.step = "0.1";
  progress.value = "0";
  progress.setAttribute("aria-label", `Seek ${label}`);

  const button = document.createElement("button");
  button.className = "carousel-play";
  button.type = "button";
  button.textContent = "Play";
  button.setAttribute("aria-label", `Play ${label}`);

  const updateProgress = () => {
    enforceSilentVideo(video);
    const percent = video.duration ? (video.currentTime / video.duration) * 100 : 0;
    progress.value = String(Math.min(100, percent));
    progress.style.setProperty("--progress", `${Math.min(100, percent)}%`);
  };

  const seekVideo = () => {
    enforceSilentVideo(video);
    if (!video.duration) return;
    video.currentTime = (Number(progress.value) / 100) * video.duration;
    updateProgress();
  };

  /* Drive the button off the media events, so a clip started by the
     comparison-pair sync still shows "Pause". */
  const syncButton = () => {
    const action = video.paused ? "Play" : "Pause";
    button.textContent = action;
    button.setAttribute("aria-label", `${action} ${label}`);
  };

  const togglePlayback = () => {
    enforceSilentVideo(video);
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  };

  video.addEventListener("timeupdate", updateProgress);
  video.addEventListener("loadedmetadata", updateProgress);
  video.addEventListener("ended", updateProgress);
  video.addEventListener("click", togglePlayback);
  video.addEventListener("play", syncButton);
  video.addEventListener("pause", syncButton);
  button.addEventListener("click", togglePlayback);
  progress.addEventListener("input", seekVideo);

  stage.append(progress, button);
  return { progress, button };
}

function createDemoCard(task) {
  const card = document.createElement("article");
  card.className = "carousel-card";
  card.dataset.platform = task.platform;
  card.dataset.orientation = task.orientation;

  const stage = document.createElement("div");
  stage.className = "carousel-video-stage";

  const label = `${task.name} inference demo`;
  const video = createSilentVideo(task.inference, label);

  const badge = document.createElement("div");
  badge.className = "video-badge";
  badge.textContent = "autonomous, 1x speed";

  stage.append(video, badge);
  attachPlayerControls(video, stage, label);

  const caption = document.createElement("p");
  caption.className = "carousel-caption";
  caption.textContent = task.name;

  card.append(stage, caption);
  return card;
}

function getTrackByPlatform(platform) {
  return document.getElementById(
    platform === "Single Arm" ? "single-arm-demo-track" : "dual-arm-demo-track",
  );
}

function scrollDemos(platform, direction) {
  const track = getTrackByPlatform(platform);
  if (!track) return;
  const card = track.querySelector(".carousel-card");
  const distance = card ? card.getBoundingClientRect().width + 14 : track.clientWidth * 0.75;
  track.scrollBy({ left: direction * distance, behavior: "smooth" });
}

function setupDemoCarousel() {
  ["Single Arm", "Dual Arm"].forEach((platform) => {
    const carousel = document.querySelector(`.demo-carousel[data-platform="${platform}"]`);
    const track = getTrackByPlatform(platform);
    if (!carousel || !track) return;

    track.replaceChildren(
      ...tasks.filter((task) => task.platform === platform).map(createDemoCard),
    );

    carousel.addEventListener("keydown", (event) => {
      /* Nested controls (seek slider, play buttons) own their arrow keys. */
      if (event.target !== carousel) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        scrollDemos(platform, -1);
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        scrollDemos(platform, 1);
      }
    });
  });

  document.querySelectorAll("[data-carousel-prev]").forEach((button) => {
    button.addEventListener("click", () => scrollDemos(button.dataset.carouselPrev, -1));
  });

  document.querySelectorAll("[data-carousel-next]").forEach((button) => {
    button.addEventListener("click", () => scrollDemos(button.dataset.carouselNext, 1));
  });
}

function setupComparisonVideos() {
  document.querySelectorAll(".comparison-video-card").forEach((card) => {
    const video = card.querySelector("video");
    if (!video) return;

    const label = video.getAttribute("aria-label") || "comparison video";
    enforceSilentVideo(video);
    video.setAttribute("muted", "");
    video.addEventListener("play", () => enforceSilentVideo(video));
    video.addEventListener("volumechange", () => enforceSilentVideo(video));

    const { progress, button } = attachPlayerControls(video, card, label);
    progress.classList.add("comparison-progress");
    button.classList.add("comparison-play");
  });
}

/* Play both clips of a comparison pair together, so VLAct and the baseline
   stay time-aligned while the visitor watches them side by side. */
function setupComparisonSync() {
  document.querySelectorAll(".comparison-pair").forEach((pair) => {
    const videos = [...pair.querySelectorAll("video")];
    if (videos.length < 2) return;

    let syncing = false;
    videos.forEach((video) => {
      video.addEventListener("play", () => {
        if (syncing) return;
        syncing = true;
        videos
          .filter((other) => other !== video && other.paused)
          .forEach((other) => {
            other.currentTime = 0;
            other.play().catch(() => {});
          });
        syncing = false;
      });
    });
  });
}

function setupScrollSpy() {
  const links = [...document.querySelectorAll(".site-nav-links a")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  if (!sections.length || !("IntersectionObserver" in window)) return;

  const visible = new Set();
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      });
      const current = sections.find((section) => visible.has(section.id));
      links.forEach((link) => {
        link.classList.toggle(
          "is-active",
          Boolean(current) && link.getAttribute("href") === `#${current.id}`,
        );
      });
    },
    { rootMargin: "-20% 0px -70% 0px", threshold: 0 },
  );

  sections.forEach((section) => observer.observe(section));
}

function setupCopyButtons() {
  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    const original = button.textContent;
    let resetTimer = null;
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch (error) {
        const range = document.createRange();
        range.selectNodeContents(target);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = "Selected";
      }
      window.clearTimeout(resetTimer);
      resetTimer = window.setTimeout(() => {
        button.textContent = original;
      }, 1600);
    });
  });
}

/* Only show the "scroll for more columns" hint on tables that actually overflow. */
function setupTableHints() {
  const sync = () => {
    document.querySelectorAll(".table-card").forEach((card) => {
      const wrap = card.querySelector(".table-wrap");
      const hint = card.querySelector(".scroll-hint");
      if (!wrap || !hint) return;
      hint.hidden = wrap.scrollWidth <= wrap.clientWidth + 2;
    });
  };
  sync();
  window.addEventListener("resize", sync);
}

setupDemoCarousel();
setupComparisonVideos();
setupComparisonSync();
setupScrollSpy();
setupCopyButtons();
setupTableHints();
