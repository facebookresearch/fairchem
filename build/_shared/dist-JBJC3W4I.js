import "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/myst-ext-button/dist/index.js
var REF_PATTERN = /^(.+?)<([^<>]+)>$/;
var buttonRole = {
  name: "button",
  doc: "Button element with an action to navigate to internal or external links.",
  body: {
    type: String,
    doc: "The body of the button.",
    required: true
  },
  run(data) {
    const body = data.body;
    const match = REF_PATTERN.exec(body);
    const [, modified, rawLabel] = match !== null && match !== void 0 ? match : [];
    const url = rawLabel !== null && rawLabel !== void 0 ? rawLabel : body;
    const node = {
      type: "link",
      url,
      children: [],
      class: "button"
      // TODO: allow users to extend this
    };
    if (modified)
      node.children = [{ type: "text", value: modified.trim() }];
    return [node];
  }
};
export {
  buttonRole
};
//# sourceMappingURL=/build/_shared/dist-JBJC3W4I.js.map
