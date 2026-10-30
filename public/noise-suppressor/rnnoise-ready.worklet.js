const register = globalThis.registerProcessor;
globalThis.registerProcessor = (name, Processor) => {
  globalThis.registerProcessor = register;
  register(name, class extends Processor {
    constructor(options) {
      super(options);
      let processor = this.processor;
      Object.defineProperty(this, "processor", {
        get: () => processor,
        set: (value) => {
          processor = value;
          if (value) this.port.postMessage("ready");
        },
      });
      if (processor) this.port.postMessage("ready");
    }
  });
};

