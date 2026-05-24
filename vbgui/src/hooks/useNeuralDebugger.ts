import { useState, useEffect, useRef } from "react";

export interface Token {
  id: number;
  text: string;
}

export function useNeuralDebugger(rpc?: any, tokenizerPath?: string | null) {
  const [debuggerMode, setDebuggerMode] = useState(false);
  const [activeStep, setActiveStep] = useState(-1); // -1 = tokenizer, 0..N-1 = layer nodes, N = loss, N+1 = de-tokenizer
  const [direction, setDirection] = useState<"forward" | "backward">("forward");
  const [prompt, setPrompt] = useState("The cat sat on the mat");
  const [tokens, setTokens] = useState<Token[]>([
    { id: 3797, text: "The" },
    { id: 3798, text: " cat" },
    { id: 3799, text: " sat" },
    { id: 3800, text: " on" },
    { id: 3801, text: " the" },
    { id: 3802, text: " mat" },
  ]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [lr, setLr] = useState(0.001);
  const [lossVal, setLossVal] = useState(2.34);
  const [isWeightUpdated, setIsWeightUpdated] = useState(false);

  // V7-F01-REAL: fetch real segmented tokens from the active tokenizer via RPC
  useEffect(() => {
    if (!rpc || !tokenizerPath || !prompt) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await rpc.call(
          "tokenizer.encode_visualize",
          { text: prompt, tokenizer_source: tokenizerPath }
        );
        if (cancelled) return;
        if (r && r.tokens) {
          setTokens(r.tokens.map((t: any) => ({ id: t.id, text: t.text })));
        }
      } catch (e) {
        console.error("Debugger tokenizer encode failed:", e);
      }
    })();
    return () => { cancelled = true; };
  }, [rpc, tokenizerPath, prompt]);

  // Use refs for intervals to prevent state staleness
  const stepTimerRef = useRef<NodeJS.Timeout | null>(null);

  const resetDebugger = () => {
    if (stepTimerRef.current) {
      clearInterval(stepTimerRef.current);
      stepTimerRef.current = null;
    }
    setActiveStep(-1);
    setDirection("forward");
    setIsPlaying(false);
    setIsWeightUpdated(false);
  };

  const stepForward = (maxStep: number) => {
    setIsWeightUpdated(false);
    if (direction === "forward") {
      setActiveStep((prev) => {
        if (prev >= maxStep) {
          // Transition to backward pass!
          setDirection("backward");
          return maxStep;
        }
        return prev + 1;
      });
    } else {
      // backward direction
      setActiveStep((prev) => {
        if (prev <= -1) {
          // Finished backward pass, reset to forward and trigger weight update pulse!
          setDirection("forward");
          setIsPlaying(false);
          setIsWeightUpdated(true);
          return -1;
        }
        return prev - 1;
      });
    }
  };

  const stepBackward = (maxStep: number) => {
    setIsWeightUpdated(false);
    if (direction === "forward") {
      setActiveStep((prev) => {
        if (prev <= -1) return -1;
        return prev - 1;
      });
    } else {
      setActiveStep((prev) => {
        if (prev >= maxStep) {
          setDirection("forward");
          return maxStep;
        }
        return prev + 1;
      });
    }
  };

  // Normal Play/Pause interval
  useEffect(() => {
    if (!isPlaying) {
      if (stepTimerRef.current) {
        clearInterval(stepTimerRef.current);
        stepTimerRef.current = null;
      }
      return;
    }

    // We calculate maxStep dynamically or assume 10 if not supplied
    // To be safe, we let interval callback execute stepForward with a threshold
    stepTimerRef.current = setInterval(() => {
      // We read activeStep and direction, so we need to be careful with closure.
      // But standard functional updates in stepForward handle state correctly!
      // Here, maxStep will be supplied from the App, but we can default to 8 if not bounded
      stepForward(8);
    }, 1200);

    return () => {
      if (stepTimerRef.current) {
        clearInterval(stepTimerRef.current);
      }
    };
  }, [isPlaying]);

  // Full animated train step sequence
  const animateFullTrainStep = (maxStep: number) => {
    resetDebugger();
    setIsPlaying(false);
    setIsWeightUpdated(false);

    let currentStep = -1;
    let currentDir: "forward" | "backward" = "forward";

    setActiveStep(currentStep);
    setDirection(currentDir);

    const intervalMs = 250; // Fast and snappy animation sequence
    stepTimerRef.current = setInterval(() => {
      if (currentDir === "forward") {
        if (currentStep >= maxStep) {
          currentDir = "backward";
          setDirection("backward");
        } else {
          currentStep += 1;
          setActiveStep(currentStep);
        }
      } else {
        if (currentStep <= -1) {
          // Complete
          clearInterval(stepTimerRef.current!);
          stepTimerRef.current = null;
          setDirection("forward");
          setIsWeightUpdated(true);
          // Gently decrease the loss to simulate learning!
          setLossVal((prev) => Math.max(0.1, prev - 0.23));
        } else {
          currentStep -= 1;
          setActiveStep(currentStep);
        }
      }
    }, intervalMs);
  };

  return {
    debuggerMode,
    setDebuggerMode,
    activeStep,
    setActiveStep,
    direction,
    setDirection,
    prompt,
    setPrompt,
    tokens,
    setTokens,
    isPlaying,
    setIsPlaying,
    lr,
    setLr,
    lossVal,
    setLossVal,
    isWeightUpdated,
    setIsWeightUpdated,
    stepForward,
    stepBackward,
    resetDebugger,
    animateFullTrainStep,
  };
}
