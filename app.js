document.documentElement.classList.add("js");

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const header = document.querySelector("[data-header]");
const progressBar = document.querySelector(".scroll-progress span");
const navToggle = document.querySelector(".nav-toggle");
const siteNav = document.querySelector(".site-nav");

const updateScrollUi = () => {
  const scrollTop = window.scrollY;
  const scrollRange = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollRange > 0 ? scrollTop / scrollRange : 0;

  header?.classList.toggle("is-scrolled", scrollTop > 18);
  if (progressBar) progressBar.style.transform = `scaleX(${progress})`;
};

let scrollTicking = false;
window.addEventListener(
  "scroll",
  () => {
    if (scrollTicking) return;
    scrollTicking = true;
    window.requestAnimationFrame(() => {
      updateScrollUi();
      scrollTicking = false;
    });
  },
  { passive: true },
);
updateScrollUi();

navToggle?.addEventListener("click", () => {
  const expanded = navToggle.getAttribute("aria-expanded") === "true";
  navToggle.setAttribute("aria-expanded", String(!expanded));
  siteNav?.classList.toggle("is-open", !expanded);
});

siteNav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navToggle?.setAttribute("aria-expanded", "false");
    siteNav.classList.remove("is-open");
  });
});

const revealElements = document.querySelectorAll(".reveal");
if (prefersReducedMotion || !("IntersectionObserver" in window)) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -7%" },
  );
  revealElements.forEach((element) => revealObserver.observe(element));
}

const countElements = document.querySelectorAll("[data-count]");
const animateCount = (element) => {
  const target = Number(element.dataset.count);
  const decimals = Number(element.dataset.decimals || 0);
  const duration = 1100;
  const startTime = performance.now();

  const step = (now) => {
    const elapsed = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    element.textContent = (target * eased).toFixed(decimals);
    if (elapsed < 1) window.requestAnimationFrame(step);
  };

  window.requestAnimationFrame(step);
};

if (!prefersReducedMotion && "IntersectionObserver" in window) {
  const countObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        animateCount(entry.target);
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.65 },
  );
  countElements.forEach((element) => countObserver.observe(element));
}

const sectionLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')];
const sections = [...document.querySelectorAll("[data-section][id]")];
if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      sectionLinks.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-28% 0px -60%", threshold: [0, 0.1, 0.4] },
  );
  sections.forEach((section) => sectionObserver.observe(section));
}

const tiltElement = document.querySelector("[data-tilt]");
if (tiltElement && !prefersReducedMotion && window.matchMedia("(pointer: fine)").matches) {
  tiltElement.addEventListener("pointermove", (event) => {
    const rect = tiltElement.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    tiltElement.style.setProperty("--tilt-x", `${x * 5}deg`);
    tiltElement.style.setProperty("--tilt-y", `${y * -5}deg`);
  });
  tiltElement.addEventListener("pointerleave", () => {
    tiltElement.style.setProperty("--tilt-x", "-2deg");
    tiltElement.style.setProperty("--tilt-y", "1deg");
  });
}

const methodTabs = [...document.querySelectorAll("[data-method]")];
const methodPanels = [...document.querySelectorAll("[data-panel]")];

const selectMethod = (name, moveFocus = false) => {
  methodTabs.forEach((tab) => {
    const selected = tab.dataset.method === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && moveFocus) tab.focus();
  });
  methodPanels.forEach((panel) => {
    const selected = panel.dataset.panel === name;
    panel.hidden = !selected;
    panel.classList.toggle("is-active", selected);
  });
};

methodTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectMethod(tab.dataset.method));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + methodTabs.length) % methodTabs.length;
    selectMethod(methodTabs[nextIndex].dataset.method, true);
  });
});
selectMethod("preserve");

const transferData = [
  { budget: 10, score: 41.42, point: [40, 132] },
  { budget: 20, score: 49.5, point: [230, 61] },
  { budget: 50, score: 51.0, point: [420, 48] },
  { budget: 100, score: 54.0, point: [610, 28] },
];

const budgetSlider = document.querySelector("#data-budget");
const budgetPercent = document.querySelector("#budget-percent");
const budgetScore = document.querySelector("#budget-score");
const budgetLabel = document.querySelector("#budget-label");
const vlactBar = document.querySelector("#vlact-bar");
const vlactBarValue = document.querySelector("#vlact-bar-value");
const activePoint = document.querySelector("#active-point");

const updateTransfer = (index) => {
  const item = transferData[index];
  if (!item) return;
  const scoreText = item.score === 41.42 ? item.score.toFixed(2) : item.score.toFixed(1);

  budgetPercent.textContent = `${item.budget}%`;
  budgetScore.textContent = `${scoreText}%`;
  budgetLabel.textContent = `${item.budget}% fine-tuning data`;
  vlactBarValue.textContent = scoreText;
  vlactBar.style.setProperty("--bar-width", `${(item.score / 60) * 100}%`);
  activePoint.setAttribute("cx", item.point[0]);
  activePoint.setAttribute("cy", item.point[1]);
};

budgetSlider?.addEventListener("input", (event) => updateTransfer(Number(event.target.value)));

const benchmarkData = {
  libero: {
    kicker: "Robust single-arm manipulation",
    number: "82.6",
    suffix: "%",
    title: "Highest score in the paper's LIBERO-Plus comparison",
    description: "VLAct improves over the matched Qwen3VL-OFT baseline by 7.6 percentage points and surpasses ABot-M0 in the reported comparison.",
    note: "Official Total across seven perturbation axes; it is not a simple mean of rounded axis values",
    barsLabel: "Official Total success rate (%)",
    decimals: 1,
    max: 100,
    bars: [
      ["VLAct", 82.6, true],
      ["ABot-M0", 80.5, false],
      ["Qwen3VL-OFT", 75.0, false],
      ["OpenVLA-OFT", 69.6, false],
    ],
  },
  vla: {
    kicker: "Behavioral generalization",
    number: "53.4",
    suffix: "%",
    title: "Highest score on every reported VLA-Arena axis",
    description: "VLAct leads on safety, distractors, extrapolation, and long-horizon behavior, improving over π0.5 by 9.1 percentage points overall.",
    note: "Official suite-weighted average across 11 task suites; category scores average L0/L1/L2",
    barsLabel: "Official weighted-average success rate (%)",
    decimals: 1,
    max: 70,
    bars: [
      ["VLAct", 53.4, true],
      ["π0.5", 44.3, false],
      ["π0", 42.3, false],
      ["OpenVLA-OFT", 39.9, false],
      ["Qwen3VL-OFT", 33.4, false],
    ],
  },
  robotwin: {
    kicker: "Bimanual manipulation under clean and randomized evaluation",
    number: "92.5",
    suffix: "%",
    title: "Highest result in the paper's Data Scaling comparison",
    description: "The same VLAct backbone reaches 92.5% on Clean and 90.8% on Random evaluation with an OFT downstream head.",
    note: "Data Scaling uses 50 clean + 500 randomized trajectories per task; 50 tasks × 100 evaluation episodes",
    barsLabel: "Data Scaling setting · Clean success rate (%)",
    decimals: 1,
    max: 100,
    bars: [
      ["VLAct", 92.5, true],
      ["LingBot-VLA", 88.6, false],
      ["Qwen3VL-OFT", 88.2, false],
      ["ABot-M0", 86.1, false],
      ["π0.5", 82.7, false],
    ],
  },
  domino: {
    kicker: "Dynamic bimanual manipulation",
    number: "18.50",
    suffix: "% SR",
    title: "Best reported result on both DOMINO metrics",
    description: "VLAct-OFT reaches 18.50% success rate and a 34.20 manipulation score, improving over the matched Qwen3VL-OFT baseline by 7.64 SR points and 3.71 MS points.",
    note: "Appendix §B.2 · one multi-task policy across all 35 tasks · clean dynamic setting",
    barsLabel: "Success Rate (%) · clean dynamic setting",
    decimals: 2,
    max: 20,
    bars: [
      ["VLAct-OFT", 18.50, true],
      ["Qwen3VL-OFT", 10.86, false],
      ["π0.5", 9.63, false],
      ["OpenVLA-OFT", 9.06, false],
      ["π0", 8.17, false],
    ],
  },
  robocasa: {
    kicker: "Held-out humanoid embodiment",
    number: "54.0",
    suffix: "%",
    title: "Full-data performance on RoboCasa-GR1",
    description: "At only 20% of RoboCasa-GR1 fine-tuning trajectories, VLAct reaches 49.5%—above the three reported 100%-data baselines shown here.",
    note: "Fine-tuned transfer · GR-1 trajectories absent from robot pre-training",
    barsLabel: "Task success rate (%)",
    decimals: 1,
    max: 60,
    bars: [
      ["VLAct", 54.0, true],
      ["Qwen3VL-OFT", 48.8, false],
      ["GR00T-N1.6", 47.6, false],
      ["π0.5", 37.0, false],
    ],
  },
  robodojo: {
    kicker: "Official simulation leaderboard",
    number: "7.60",
    suffix: "%",
    title: "Sixth by success rate; eighth by score",
    description: "VLAct records a 10.66 average partial-progress score and 7.60% success, exceeding every explicitly designated WAM entry on both aggregate metrics.",
    note: "Selected entries · 42 tasks × 50 episodes · official 35-policy snapshot, August 24, 2026",
    barsLabel: "Average success rate (%) · selected entries",
    decimals: 2,
    max: 10,
    bars: [
      ["VLAct", 7.60, true],
      ["InternVLA-A1.5", 7.14, false],
      ["π0.5", 6.91, false],
      ["X-WAM", 3.83, false],
    ],
  },
};

const benchmarkTabs = [...document.querySelectorAll("[data-benchmark]")];
const benchmarkKicker = document.querySelector("#benchmark-kicker");
const benchmarkNumber = document.querySelector("#benchmark-number");
const benchmarkTitle = document.querySelector("#benchmark-title");
const benchmarkDescription = document.querySelector("#benchmark-description");
const benchmarkNote = document.querySelector("#benchmark-note");
const benchmarkBarsLabel = document.querySelector("#benchmark-bars-label");
const benchmarkBars = document.querySelector("#benchmark-bars");
const benchmarkContent = document.querySelector("#benchmark-content");

const renderBenchmark = (name) => {
  const data = benchmarkData[name];
  if (!data) return;

  benchmarkTabs.forEach((tab) => {
    const selected = tab.dataset.benchmark === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });

  benchmarkKicker.textContent = data.kicker;
  benchmarkNumber.innerHTML = `${data.number}<span>${data.suffix}</span>`;
  benchmarkTitle.textContent = data.title;
  benchmarkDescription.textContent = data.description;
  benchmarkNote.textContent = data.note;
  benchmarkBarsLabel.textContent = data.barsLabel;
  const selectedTab = benchmarkTabs.find((tab) => tab.dataset.benchmark === name);
  if (selectedTab && benchmarkContent) benchmarkContent.setAttribute("aria-labelledby", selectedTab.id);
  benchmarkBars.innerHTML = "";

  data.bars.forEach(([label, value, primary]) => {
    const row = document.createElement("div");
    row.className = `benchmark-bar-row${primary ? " is-primary" : ""}`;

    const nameElement = document.createElement("span");
    nameElement.textContent = label;
    const track = document.createElement("div");
    track.className = "benchmark-bar-track";
    const fill = document.createElement("div");
    fill.className = "benchmark-bar-fill";
    track.appendChild(fill);
    const valueElement = document.createElement("strong");
    valueElement.textContent = Number(value).toFixed(data.decimals);

    row.append(nameElement, track, valueElement);
    benchmarkBars.appendChild(row);
    window.requestAnimationFrame(() => {
      fill.style.width = `${Math.min((value / data.max) * 100, 100)}%`;
    });
  });
};

benchmarkTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => renderBenchmark(tab.dataset.benchmark));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + benchmarkTabs.length) % benchmarkTabs.length;
    renderBenchmark(benchmarkTabs[nextIndex].dataset.benchmark);
    benchmarkTabs[nextIndex].focus();
  });
});
renderBenchmark("libero");

const realData = {
  short: {
    number: "92.5%",
    title: "Reliable short-horizon control",
    description: "Mean success across four in-domain single-arm tasks, compared with 77.5% for Qwen3VL-4B-OFT without VLA pre-training; 10 rollouts per task.",
    gain: "+15.0 pp",
  },
  novel: {
    number: "90.0%",
    title: "Objects change. The behavior holds.",
    description: "Separate averages for the novel-object-from-pot and novel-object-in-cup task families are both 90.0%; Qwen3VL-4B-OFT without VLA pre-training reaches 73.3% and 65.0%.",
    gain: "+16.7 / +25.0 pp",
  },
  long: {
    number: "86.6% / 80.0%",
    title: "Longer tasks retain their logic",
    description: "Stage-weighted completion rates for table cleaning and scooping beans, versus 73.3% and 33.3% for Qwen3VL-4B-OFT without VLA pre-training.",
    gain: "+13.3 / +46.7 pp",
  },
  dual: {
    number: "72.0%",
    title: "Transfer to dual-arm coordination",
    description: "Mean success across five real-world dual-arm Franka tasks, versus 44.0% for Qwen3VL-4B-OFT without VLA pre-training; 10 rollouts per task.",
    gain: "+28.0 pp",
  },
};

const realTabs = [...document.querySelectorAll("[data-real]")];
const realNumber = document.querySelector("#real-number");
const realTitle = document.querySelector("#real-title");
const realDescription = document.querySelector("#real-description");
const realGain = document.querySelector("#real-gain");
const realResultPanel = document.querySelector("#real-result");

const renderRealResult = (name) => {
  const data = realData[name];
  if (!data) return;
  realTabs.forEach((tab) => {
    const selected = tab.dataset.real === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  realNumber.textContent = data.number;
  realNumber.classList.toggle("is-long", data.number.length > 8);
  realTitle.textContent = data.title;
  realDescription.textContent = data.description;
  realGain.textContent = data.gain;
  const selectedTab = realTabs.find((tab) => tab.dataset.real === name);
  if (selectedTab && realResultPanel) realResultPanel.setAttribute("aria-labelledby", selectedTab.id);
};

realTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => renderRealResult(tab.dataset.real));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + realTabs.length) % realTabs.length;
    renderRealResult(realTabs[nextIndex].dataset.real);
    realTabs[nextIndex].focus();
  });
});
renderRealResult("short");

const rolloutData = {
  cube: {
    label: "Cube stacking",
    type: "VLAct · single arm, short horizon",
    src: "assets/videos/stack-blocks.mp4",
    poster: "assets/video-posters/stack-blocks.jpg",
    layout: "portrait",
    ariaLabel: "VLAct cube-stacking real-robot rollout",
  },
  carrot: {
    label: "Carrot-from-pot",
    type: "VLAct · single arm, short horizon",
    src: "assets/videos/take-carrot-from-pot-cropped.mp4",
    poster: "assets/video-posters/take-carrot-from-pot.jpg",
    layout: "portrait",
    ariaLabel: "VLAct carrot-from-pot real-robot rollout",
  },
  button: {
    label: "Button pressing",
    type: "VLAct · single arm, short horizon",
    src: "assets/videos/press-button.mp4",
    poster: "assets/video-posters/press-button.jpg",
    layout: "portrait",
    ariaLabel: "VLAct button-pressing real-robot rollout",
  },
  pen: {
    label: "Pen in cup",
    type: "VLAct · single arm, short horizon",
    src: "assets/videos/place-pen-into-cup.mp4",
    poster: "assets/video-posters/place-pen-into-cup.jpg",
    layout: "portrait",
    ariaLabel: "VLAct pen-in-cup real-robot rollout",
  },
  table: {
    label: "Table cleaning",
    type: "VLAct · single arm, long horizon",
    src: "assets/videos/clean-table-cropped.mp4",
    poster: "assets/video-posters/clean-table.jpg",
    layout: "portrait",
    ariaLabel: "VLAct table-cleaning real-robot rollout",
  },
  beans: {
    label: "Scoop beans",
    type: "VLAct · single arm, long horizon",
    src: "assets/videos/scoop-beans-cropped.mp4",
    poster: "assets/video-posters/scoop-beans.jpg",
    layout: "portrait",
    ariaLabel: "VLAct scoop-beans real-robot rollout",
  },
  unplug: {
    label: "Unplugging",
    type: "VLAct · dual-arm coordination",
    src: "assets/videos/unplug-cable.mp4",
    poster: "assets/video-posters/unplug-cable.jpg",
    layout: "landscape",
    ariaLabel: "VLAct dual-arm unplugging real-robot rollout",
  },
  breakfast: {
    label: "Breakfast preparation",
    type: "VLAct · dual-arm coordination",
    src: "assets/videos/prepare-breakfast.mp4",
    poster: "assets/video-posters/prepare-breakfast.jpg",
    layout: "landscape",
    ariaLabel: "VLAct dual-arm breakfast-preparation real-robot rollout",
  },
  handover: {
    label: "Banana handover-place",
    type: "VLAct · dual-arm coordination",
    src: "assets/videos/handover-banana.mp4",
    poster: "assets/video-posters/handover-banana.jpg",
    layout: "landscape",
    ariaLabel: "VLAct dual-arm banana-handover-place real-robot rollout",
  },
  pants: {
    label: "Fold pants",
    type: "VLAct · dual-arm coordination",
    src: "assets/videos/fold-pants.mp4",
    poster: "assets/video-posters/fold-pants.jpg",
    layout: "landscape",
    ariaLabel: "VLAct dual-arm fold-pants real-robot rollout",
  },
  towel: {
    label: "Fold towel",
    type: "VLAct · dual-arm coordination",
    src: "assets/videos/fold-towel.mp4",
    poster: "assets/video-posters/fold-towel.jpg",
    layout: "landscape",
    ariaLabel: "VLAct dual-arm fold-towel real-robot rollout",
  },
  "table-baseline": {
    label: "Table cleaning — baseline",
    type: "Qwen3VL-4B-OFT without pre-training",
    src: "assets/videos/clean-table-compare.mp4",
    poster: "assets/video-posters/clean-table-compare.jpg",
    layout: "portrait",
    ariaLabel: "Qwen3VL-4B-OFT table-cleaning baseline real-robot rollout",
  },
  "beans-baseline": {
    label: "Scoop beans — baseline",
    type: "Qwen3VL-4B-OFT without pre-training",
    src: "assets/videos/scoop-beans-compare.mp4",
    poster: "assets/video-posters/scoop-beans-compare.jpg",
    layout: "portrait",
    ariaLabel: "Qwen3VL-4B-OFT scoop-beans baseline real-robot rollout",
  },
};

const rolloutStage = document.querySelector("[data-rollout-stage]");
const rolloutVideo = document.querySelector("#rollout-video");
const rolloutLabel = document.querySelector("#rollout-label");
const rolloutType = document.querySelector("#rollout-type");
const rolloutCounter = document.querySelector("#rollout-counter");
const rolloutStatus = document.querySelector("#rollout-video-status");
const rolloutError = document.querySelector("#rollout-error");
const rolloutErrorCopy = document.querySelector("#rollout-error-copy");
const rolloutRetry = document.querySelector("#rollout-retry");
const rolloutTaskButtons = [...document.querySelectorAll("[data-rollout]")];
let rolloutTask = "cube";
let rolloutGeneration = 0;
let rolloutExpectedSrc = new URL(rolloutData.cube.src, document.baseURI).href;
let rolloutInViewport = false;
let rolloutDocumentVisible = document.visibilityState === "visible";
let rolloutUserWantsPlayback = !prefersReducedMotion;
let rolloutPolicyPauseGeneration = null;
let rolloutSwitching = false;
let rolloutLastUserInteraction = -Infinity;

if (rolloutVideo) {
  rolloutVideo.muted = true;
  rolloutVideo.defaultMuted = true;
  rolloutVideo.disablePictureInPicture = true;
  rolloutVideo.disableRemotePlayback = true;
}

const activeRolloutSource = () => rolloutVideo?.currentSrc || rolloutVideo?.src || "";

const isExpectedRolloutSource = () => {
  const activeSource = activeRolloutSource();
  return !activeSource || activeSource === rolloutExpectedSrc;
};

const setRolloutLoading = (isLoading) => {
  rolloutStage?.classList.toggle("is-loading", isLoading);
  if (isLoading) rolloutStage?.setAttribute("aria-busy", "true");
  else rolloutStage?.removeAttribute("aria-busy");
};

const clearRolloutError = () => {
  rolloutStage?.classList.remove("is-error");
  if (rolloutError) rolloutError.hidden = true;
};

const showRolloutError = () => {
  setRolloutLoading(false);
  rolloutStage?.classList.add("is-error");
  if (rolloutErrorCopy) {
    rolloutErrorCopy.textContent = `${rolloutData[rolloutTask].label} could not be played. Try again or choose another rollout.`;
  }
  if (rolloutError) rolloutError.hidden = false;
};

const pauseRolloutProgrammatically = () => {
  if (!rolloutVideo || rolloutVideo.paused) return;
  rolloutPolicyPauseGeneration = rolloutGeneration;
  rolloutVideo.pause();
};

const syncRolloutPlayback = (generation = rolloutGeneration) => {
  if (!rolloutVideo || generation !== rolloutGeneration) return;
  const shouldPlay = rolloutUserWantsPlayback && rolloutInViewport && rolloutDocumentVisible;

  if (!shouldPlay) {
    pauseRolloutProgrammatically();
    return;
  }

  rolloutVideo.muted = true;
  const playRequest = rolloutVideo.play();
  const requestedSource = rolloutExpectedSrc;
  playRequest?.catch((error) => {
    if (
      generation !== rolloutGeneration
      || requestedSource !== rolloutExpectedSrc
      || error?.name === "AbortError"
    ) return;

    setRolloutLoading(false);
    if (error?.name === "NotAllowedError") {
      rolloutUserWantsPlayback = false;
      if (rolloutStatus) {
        rolloutStatus.textContent = `${rolloutData[rolloutTask].label} is ready. Press play to begin.`;
      }
      return;
    }

    if (rolloutStatus) {
      rolloutStatus.textContent = `${rolloutData[rolloutTask].label} could not start playing.`;
    }
  });
};

const applyRolloutPresentation = (name) => {
  const task = rolloutData[name];
  if (!task || !rolloutVideo) return;

  rolloutTask = name;
  rolloutLabel.textContent = task.label;
  rolloutType.textContent = task.type;
  rolloutVideo.setAttribute("aria-label", task.ariaLabel);
  rolloutVideo.poster = task.poster;
  rolloutStage?.classList.toggle("is-portrait", task.layout === "portrait");
  rolloutStage?.style.setProperty("--rollout-poster", `url("${task.poster}")`);

  rolloutTaskButtons.forEach((button) => {
    const selected = button.dataset.rollout === name;
    button.setAttribute("aria-pressed", String(selected));
  });

  const selectedIndex = rolloutTaskButtons.findIndex((button) => button.dataset.rollout === name);
  if (rolloutCounter && selectedIndex >= 0) {
    rolloutCounter.textContent = `${String(selectedIndex + 1).padStart(2, "0")} / ${String(rolloutTaskButtons.length).padStart(2, "0")}`;
  }
};

const selectRollout = (
  name,
  { announce = true, userInitiated = true, force = false } = {},
) => {
  const task = rolloutData[name];
  if (!task || !rolloutVideo) return;

  if (userInitiated) {
    rolloutUserWantsPlayback = true;
    rolloutLastUserInteraction = -Infinity;
  }
  applyRolloutPresentation(name);
  clearRolloutError();

  const requestedSrc = new URL(task.src, document.baseURI).href;
  const configuredSrc = rolloutVideo.getAttribute("src");
  const configuredUrl = configuredSrc ? new URL(configuredSrc, document.baseURI).href : "";

  rolloutExpectedSrc = requestedSrc;
  const needsReload = force || Boolean(rolloutVideo.error) || configuredUrl !== requestedSrc;
  if (!needsReload) {
    rolloutSwitching = false;
    setRolloutLoading(rolloutVideo.readyState < HTMLMediaElement.HAVE_METADATA);
    if (announce && rolloutStatus) rolloutStatus.textContent = `${task.label} selected.`;
    syncRolloutPlayback();
    return;
  }

  rolloutGeneration += 1;
  const generation = rolloutGeneration;
  rolloutSwitching = true;
  rolloutPolicyPauseGeneration = null;
  setRolloutLoading(true);
  rolloutVideo.src = task.src;
  rolloutVideo.load();

  if (announce && rolloutStatus) rolloutStatus.textContent = `Loading ${task.label} video.`;
  syncRolloutPlayback(generation);
};

rolloutVideo?.addEventListener("loadedmetadata", () => {
  if (!isExpectedRolloutSource()) return;
  rolloutSwitching = false;
  setRolloutLoading(false);
  if (rolloutStatus) rolloutStatus.textContent = `${rolloutData[rolloutTask].label} video loaded.`;
  syncRolloutPlayback();
});

rolloutVideo?.addEventListener("loadeddata", () => {
  if (!isExpectedRolloutSource()) return;
  setRolloutLoading(false);
});

rolloutVideo?.addEventListener("canplay", () => {
  if (!isExpectedRolloutSource()) return;
  setRolloutLoading(false);
  syncRolloutPlayback();
});

rolloutVideo?.addEventListener("playing", () => {
  if (!isExpectedRolloutSource()) return;
  rolloutSwitching = false;
  clearRolloutError();
  setRolloutLoading(false);
});

rolloutVideo?.addEventListener("waiting", () => {
  if (isExpectedRolloutSource() && rolloutUserWantsPlayback) setRolloutLoading(true);
});

rolloutVideo?.addEventListener("seeking", () => {
  if (isExpectedRolloutSource()) setRolloutLoading(true);
});

rolloutVideo?.addEventListener("seeked", () => {
  if (!isExpectedRolloutSource()) return;
  setRolloutLoading(false);
  syncRolloutPlayback();
});

rolloutVideo?.addEventListener("error", () => {
  if (!isExpectedRolloutSource()) return;
  rolloutSwitching = false;
  rolloutPolicyPauseGeneration = null;
  showRolloutError();
  if (rolloutStatus) rolloutStatus.textContent = `${rolloutData[rolloutTask].label} video could not be loaded.`;
});

rolloutVideo?.addEventListener("play", () => {
  rolloutUserWantsPlayback = true;
});

rolloutVideo?.addEventListener("pointerdown", () => {
  rolloutLastUserInteraction = performance.now();
}, { passive: true });

rolloutVideo?.addEventListener("touchstart", () => {
  rolloutLastUserInteraction = performance.now();
}, { passive: true });

rolloutVideo?.addEventListener("keydown", () => {
  rolloutLastUserInteraction = performance.now();
});

rolloutVideo?.addEventListener("pause", () => {
  if (rolloutPolicyPauseGeneration === rolloutGeneration) {
    rolloutPolicyPauseGeneration = null;
    return;
  }
  const followsUserInteraction = performance.now() - rolloutLastUserInteraction < 1500;
  if (followsUserInteraction) {
    rolloutUserWantsPlayback = false;
    setRolloutLoading(false);
    return;
  }
  if (!rolloutSwitching) setRolloutLoading(false);
});

rolloutTaskButtons.forEach((button, index) => {
  button.addEventListener("click", () => selectRollout(button.dataset.rollout));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const nextIndex = event.key === "Home"
      ? 0
      : event.key === "End"
        ? rolloutTaskButtons.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + rolloutTaskButtons.length) % rolloutTaskButtons.length;
    const nextButton = rolloutTaskButtons[nextIndex];
    selectRollout(nextButton.dataset.rollout);
    nextButton.focus();
    nextButton.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "nearest",
      inline: "center",
    });
  });
});

if (rolloutStage && "IntersectionObserver" in window) {
  const rolloutObserver = new IntersectionObserver(
    ([entry]) => {
      rolloutInViewport = entry.isIntersecting && entry.intersectionRatio >= 0.2;
      syncRolloutPlayback();
    },
    { threshold: [0, 0.2] },
  );
  rolloutObserver.observe(rolloutStage);
} else if (rolloutStage) {
  let rolloutViewportTicking = false;
  const updateRolloutViewport = () => {
    const bounds = rolloutStage.getBoundingClientRect();
    const visibleHeight = Math.max(0, Math.min(bounds.bottom, window.innerHeight) - Math.max(bounds.top, 0));
    rolloutInViewport = bounds.height > 0 && visibleHeight / bounds.height >= 0.2;
    rolloutViewportTicking = false;
    syncRolloutPlayback();
  };
  const requestRolloutViewportUpdate = () => {
    if (rolloutViewportTicking) return;
    rolloutViewportTicking = true;
    window.requestAnimationFrame(updateRolloutViewport);
  };
  window.addEventListener("scroll", requestRolloutViewportUpdate, { passive: true });
  window.addEventListener("resize", requestRolloutViewportUpdate, { passive: true });
  requestRolloutViewportUpdate();
}

document.addEventListener("visibilitychange", () => {
  rolloutDocumentVisible = document.visibilityState === "visible";
  syncRolloutPlayback();
});

rolloutRetry?.addEventListener("click", () => {
  selectRollout(rolloutTask, { announce: true, userInitiated: true, force: true });
});

selectRollout("cube", { announce: false, userInitiated: false });

const citationButton = document.querySelector("#copy-citation");
const citationText = document.querySelector("#citation-text");

const copyCitation = async () => {
  const text = citationText?.innerText || "";
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citationText);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }
  citationButton.textContent = "Copied";
  window.setTimeout(() => {
    citationButton.textContent = "Copy citation";
  }, 1800);
};

citationButton?.addEventListener("click", copyCitation);

const modal = document.querySelector("#figure-modal");
const modalImage = document.querySelector("#modal-image");
const modalTitle = document.querySelector("#modal-title");

document.querySelectorAll("[data-zoom]").forEach((button) => {
  button.addEventListener("click", () => {
    modalImage.src = button.dataset.zoom;
    modalImage.alt = button.dataset.zoomTitle || "Expanded research figure";
    modalTitle.textContent = button.dataset.zoomTitle || "Research figure";
    if (typeof modal.showModal === "function") modal.showModal();
  });
});

modal?.querySelector(".modal-close")?.addEventListener("click", () => modal.close());
modal?.addEventListener("click", (event) => {
  if (event.target === modal) modal.close();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && modal?.open) modal.close();
});

const currentYear = document.querySelector("#current-year");
if (currentYear) currentYear.textContent = String(new Date().getFullYear());
