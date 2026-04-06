import {
  __name
} from "/build/_shared/chunk-7PDOACA3.js";

// ../../node_modules/mermaid/dist/chunks/mermaid.core/chunk-QZHKN3VN.mjs
var ImperativeState = class {
  /**
   * @param init - Function that creates the default state.
   */
  constructor(init) {
    this.init = init;
    this.records = this.init();
  }
  static {
    __name(this, "ImperativeState");
  }
  reset() {
    this.records = this.init();
  }
};

export {
  ImperativeState
};
//# sourceMappingURL=/build/_shared/chunk-TNPPA6Y6.js.map
