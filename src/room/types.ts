
interface RoomConfig {
    dimensions: {
        width: number;
        height: number;
        depth: number;
    };
    materials: {
        walls: { absorption: number };
        ceiling: { absorption: number };
        floor: { absorption: number };
    };
}

interface RoomDimensions {
    width: number;
    height: number;
    depth: number;
}

interface RoomMaterials {
    walls: { absorption: number };
    ceiling: { absorption: number };
    floor: { absorption: number };
}

export enum Surface {
    FLOOR = 0,
    CEILING = 1,
    WALL_FRONT_BACK = 2,
    WALL_LEFT_RIGHT = 3
}