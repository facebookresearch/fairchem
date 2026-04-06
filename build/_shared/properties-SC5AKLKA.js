import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/properties.js
var properties;
var init_properties = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/properties.js"() {
    properties = {
      name: "properties",
      token: function(stream, state) {
        var sol = stream.sol() || state.afterSection;
        var eol = stream.eol();
        state.afterSection = false;
        if (sol) {
          if (state.nextMultiline) {
            state.inMultiline = true;
            state.nextMultiline = false;
          } else {
            state.position = "def";
          }
        }
        if (eol && !state.nextMultiline) {
          state.inMultiline = false;
          state.position = "def";
        }
        if (sol) {
          while (stream.eatSpace()) {
          }
        }
        var ch = stream.next();
        if (sol && (ch === "#" || ch === "!" || ch === ";")) {
          state.position = "comment";
          stream.skipToEnd();
          return "comment";
        } else if (sol && ch === "[") {
          state.afterSection = true;
          stream.skipTo("]");
          stream.eat("]");
          return "header";
        } else if (ch === "=" || ch === ":") {
          state.position = "quote";
          return null;
        } else if (ch === "\\" && state.position === "quote") {
          if (stream.eol()) {
            state.nextMultiline = true;
          }
        }
        return state.position;
      },
      startState: function() {
        return {
          position: "def",
          // Current position, "def", "quote" or "comment"
          nextMultiline: false,
          // Is the next line multiline value
          inMultiline: false,
          // Is the current line a multiline value
          afterSection: false
          // Did we just open a section
        };
      }
    };
  }
});
init_properties();
export {
  properties
};
//# sourceMappingURL=/build/_shared/properties-SC5AKLKA.js.map
